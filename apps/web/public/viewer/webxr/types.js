/**
 * Minimal WebXR Device API types.
 *
 * Mirrors the subset of the WebXR Device API
 * (https://www.w3.org/TR/webxr/) consumed by {@link WebXRSplatViewer} and
 * its unit tests. We declare these locally instead of pulling
 * `@types/webxr` so the viewer package keeps zero runtime deps and so the
 * Vitest mocks can plug in a structurally-typed stub.
 *
 * The shapes match the live browser API one-for-one — `XRView.transform`
 * has a row-major `matrix: Float32Array` of length 16, `XRRigidTransform`
 * exposes `position`/`orientation` DOMPointReadOnlys, etc.
 */
export {};
//# sourceMappingURL=types.js.map