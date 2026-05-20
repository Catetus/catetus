/**
 * Minimal typed event emitter. No external deps.
 *
 * Generic parameter `T` is an object whose keys are event names and whose
 * values are the payload type for that event.
 */
export type EventMap = Record<string, unknown>;

type Listener<P> = (payload: P) => void;

/**
 * Tiny typed pub-sub used by {@link CatetusViewer}.
 *
 * @example
 * ```ts
 * const e = new Emitter<{ tick: { n: number } }>();
 * e.on('tick', ({ n }) => console.log(n));
 * e.emit('tick', { n: 1 });
 * ```
 */
export class Emitter<T extends EventMap> {
  private readonly listeners: Map<keyof T, Set<Listener<unknown>>> = new Map();

  /** Subscribe to an event. Returns an unsubscribe function. */
  on<K extends keyof T>(event: K, fn: Listener<T[K]>): () => void {
    let set = this.listeners.get(event);
    if (!set) {
      set = new Set();
      this.listeners.set(event, set);
    }
    set.add(fn as Listener<unknown>);
    return () => this.off(event, fn);
  }

  /** Unsubscribe a previously registered listener. */
  off<K extends keyof T>(event: K, fn: Listener<T[K]>): void {
    const set = this.listeners.get(event);
    if (!set) return;
    set.delete(fn as Listener<unknown>);
    if (set.size === 0) this.listeners.delete(event);
  }

  /** Subscribe for exactly one delivery, then auto-unsubscribe. */
  once<K extends keyof T>(event: K, fn: Listener<T[K]>): () => void {
    const wrapped: Listener<T[K]> = (payload) => {
      this.off(event, wrapped);
      fn(payload);
    };
    return this.on(event, wrapped);
  }

  /** Synchronously fan out `payload` to all listeners of `event`. */
  emit<K extends keyof T>(event: K, payload: T[K]): void {
    const set = this.listeners.get(event);
    if (!set || set.size === 0) return;
    // Snapshot listeners so removal during iteration is safe.
    for (const fn of [...set]) {
      try {
        (fn as Listener<T[K]>)(payload);
      } catch {
        // Listeners must not be able to break the emit loop.
      }
    }
  }

  /** Drop every listener. Called by {@link CatetusViewer.dispose}. */
  removeAll(): void {
    this.listeners.clear();
  }
}
