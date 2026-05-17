// Ambient declarations for hero-only third-party libs.
// We use mkkellogg/GaussianSplats3D for the marketing hero only; full
// types from @types/three aren't pulled in here because the SDK we ship
// (packages/viewer) deliberately doesn't depend on Three.js.
declare module "@mkkellogg/gaussian-splats-3d";
declare module "three";
