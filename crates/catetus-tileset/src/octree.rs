//! Depth-limited spatial octree over splat positions.
//!
//! Each node owns an axis-aligned bounding box (computed from its splats) and
//! either children (internal) or a set of splat indices (leaf). Subdivision
//! stops when a node has <= `max_splats_per_leaf` splats OR reaches
//! `max_depth`. Empty octants are pruned so the tree is sparse — exactly the
//! shape SuperSplat's `lod-meta.json` tree has (interior nodes with 1..8
//! children, never the full 8 unless the scene fills them).

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Bounds {
    /// An empty (inverted) box that swallows points via [`Bounds::expand`].
    pub fn empty() -> Self {
        Self { min: [f32::MAX; 3], max: [f32::MIN; 3] }
    }

    /// Grow the box to include `p`.
    pub fn expand(&mut self, p: [f32; 3]) {
        for k in 0..3 {
            if p[k] < self.min[k] {
                self.min[k] = p[k];
            }
            if p[k] > self.max[k] {
                self.max[k] = p[k];
            }
        }
    }

    /// Center point of the box. Returns origin for an empty box.
    pub fn center(&self) -> [f32; 3] {
        if self.min[0] > self.max[0] {
            return [0.0; 3];
        }
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Length of the box diagonal — used as the node's geometric error proxy.
    pub fn diagonal(&self) -> f32 {
        if self.min[0] > self.max[0] {
            return 0.0;
        }
        let dx = self.max[0] - self.min[0];
        let dy = self.max[1] - self.min[1];
        let dz = self.max[2] - self.min[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Octree build parameters.
#[derive(Debug, Clone, Copy)]
pub struct OctreeConfig {
    /// Maximum subdivision depth (root = depth 0). SuperSplat's Koriyama tree
    /// is depth 6 (7 levels). Default 6.
    pub max_depth: usize,
    /// A node with this many or fewer splats becomes a leaf.
    pub max_splats_per_leaf: usize,
}

impl Default for OctreeConfig {
    fn default() -> Self {
        Self { max_depth: 6, max_splats_per_leaf: 50_000 }
    }
}

/// A node in the octree. **Leaves** carry their full `splat_indices` (bounded
/// by `max_splats_per_leaf`); **internal** nodes carry an EMPTY `splat_indices`
/// and their `children`. `bounds` always reflects the actual splats beneath the
/// node.
///
/// Memory: this is the STREAM-5 invariant. Previously every node retained the
/// full set of indices beneath it, so final-tree memory was
/// `Σ_nodes |indices| ≈ N·(depth+1)` — ~1.3 GB of indices for a 48M-splat
/// 7-level tree, which OOM'd at depth ~7. Now indices live ONLY on leaves
/// (each splat in exactly one leaf), so the resident index memory is `O(N)`,
/// independent of depth. The coarse per-node proxies that internal nodes need
/// for streaming are built **bottom-up** during planning (see `plan.rs`), each
/// bounded by a fixed `proxy_cap`, rather than re-derived from a full
/// per-node index set.
#[derive(Debug, Clone)]
pub struct OctreeNode {
    pub bounds: Bounds,
    pub depth: usize,
    /// Splat indices owned by this node. Non-empty ONLY for leaves; internal
    /// nodes leave this empty (their splats are partitioned among descendants
    /// and ultimately land in leaves). Conservation invariant: the union of
    /// all leaves' `splat_indices` is exactly the input set, each splat once.
    pub splat_indices: Vec<u32>,
    pub children: Vec<OctreeNode>,
}

impl OctreeNode {
    /// True if this node has no children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// A built octree.
#[derive(Debug, Clone)]
pub struct Octree {
    pub root: OctreeNode,
    pub config: OctreeConfig,
    /// Number of nodes in the tree (cached).
    pub node_count: usize,
    /// Max depth actually reached (0-based).
    pub depth_reached: usize,
}

impl Octree {
    /// Build an octree from splat positions.
    ///
    /// `positions[i]` is the world-space position of splat `i`. The returned
    /// tree references splats by their original index, so the caller can pull
    /// any attribute it likes from the original cloud.
    pub fn build(positions: &[[f32; 3]], config: OctreeConfig) -> Self {
        let mut root_bounds = Bounds::empty();
        for p in positions {
            root_bounds.expand(*p);
        }
        let all: Vec<u32> = (0..positions.len() as u32).collect();
        let mut node_count = 0usize;
        let mut depth_reached = 0usize;
        let root = Self::subdivide(
            positions,
            all,
            root_bounds,
            0,
            &config,
            &mut node_count,
            &mut depth_reached,
        );
        Self { root, config, node_count, depth_reached }
    }

    fn subdivide(
        positions: &[[f32; 3]],
        indices: Vec<u32>,
        bounds: Bounds,
        depth: usize,
        config: &OctreeConfig,
        node_count: &mut usize,
        depth_reached: &mut usize,
    ) -> OctreeNode {
        *node_count += 1;
        if depth > *depth_reached {
            *depth_reached = depth;
        }

        // Stop condition: small enough or too deep -> leaf.
        if indices.len() <= config.max_splats_per_leaf || depth >= config.max_depth {
            return OctreeNode { bounds, depth, splat_indices: indices, children: Vec::new() };
        }

        // Split at the box center into up to 8 octants.
        let c = bounds.center();
        let mut buckets: [Vec<u32>; 8] = Default::default();
        for &i in &indices {
            let p = positions[i as usize];
            let octant = ((p[0] >= c[0]) as usize)
                | (((p[1] >= c[1]) as usize) << 1)
                | (((p[2] >= c[2]) as usize) << 2);
            buckets[octant].push(i);
        }

        // Degenerate split (everything in one octant): all points coincide or
        // are collinear at this scale. Force a leaf to avoid infinite recursion.
        let nonempty = buckets.iter().filter(|b| !b.is_empty()).count();
        if nonempty <= 1 {
            return OctreeNode { bounds, depth, splat_indices: indices, children: Vec::new() };
        }

        // `indices` is dropped here: an internal node does NOT retain the full
        // subtree index set (that was the STREAM-5 OOM). Each bucket is moved
        // into its child; what survives in the final tree is only the leaves'
        // indices.
        drop(indices);

        let mut children = Vec::with_capacity(nonempty);
        for bucket in buckets {
            if bucket.is_empty() {
                continue;
            }
            let mut child_bounds = Bounds::empty();
            for &i in &bucket {
                child_bounds.expand(positions[i as usize]);
            }
            children.push(Self::subdivide(
                positions,
                bucket,
                child_bounds,
                depth + 1,
                config,
                node_count,
                depth_reached,
            ));
        }

        OctreeNode { bounds, depth, splat_indices: Vec::new(), children }
    }

    /// Count leaves (for diagnostics / conservation checks).
    pub fn leaf_count(&self) -> usize {
        fn rec(n: &OctreeNode) -> usize {
            if n.is_leaf() {
                1
            } else {
                n.children.iter().map(rec).sum()
            }
        }
        rec(&self.root)
    }

    /// Sum of splat indices held across all *leaves* — should equal the input
    /// splat count exactly (each splat lands in exactly one leaf).
    pub fn leaf_index_total(&self) -> usize {
        fn rec(n: &OctreeNode) -> usize {
            if n.is_leaf() {
                n.splat_indices.len()
            } else {
                n.children.iter().map(rec).sum()
            }
        }
        rec(&self.root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_positions(n: usize) -> Vec<[f32; 3]> {
        // n^3 points on a regular grid in [0,1]^3.
        let mut v = Vec::new();
        for x in 0..n {
            for y in 0..n {
                for z in 0..n {
                    v.push([
                        x as f32 / n as f32,
                        y as f32 / n as f32,
                        z as f32 / n as f32,
                    ]);
                }
            }
        }
        v
    }

    #[test]
    fn conservation_each_splat_in_one_leaf() {
        let pos = grid_positions(16); // 4096 points
        let tree = Octree::build(&pos, OctreeConfig { max_depth: 6, max_splats_per_leaf: 64 });
        assert_eq!(tree.leaf_index_total(), pos.len());
    }

    #[test]
    fn coincident_points_terminate() {
        // 1000 identical points must not blow the stack.
        let pos = vec![[1.0f32, 2.0, 3.0]; 1000];
        let tree = Octree::build(&pos, OctreeConfig { max_depth: 8, max_splats_per_leaf: 10 });
        assert_eq!(tree.leaf_index_total(), 1000);
        assert!(tree.root.is_leaf());
    }

    #[test]
    fn subdivides_when_dense() {
        let pos = grid_positions(8); // 512 points
        let tree = Octree::build(&pos, OctreeConfig { max_depth: 6, max_splats_per_leaf: 32 });
        assert!(tree.leaf_count() > 1, "dense scene should subdivide");
        assert!(tree.depth_reached >= 1);
    }

    #[test]
    fn bounds_diagonal_decreases_with_depth() {
        let pos = grid_positions(8);
        let tree = Octree::build(&pos, OctreeConfig { max_depth: 4, max_splats_per_leaf: 16 });
        let root_diag = tree.root.bounds.diagonal();
        for child in &tree.root.children {
            assert!(child.bounds.diagonal() <= root_diag + 1e-6);
        }
    }
}
