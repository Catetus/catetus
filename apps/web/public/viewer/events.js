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
export class Emitter {
    listeners = new Map();
    /** Subscribe to an event. Returns an unsubscribe function. */
    on(event, fn) {
        let set = this.listeners.get(event);
        if (!set) {
            set = new Set();
            this.listeners.set(event, set);
        }
        set.add(fn);
        return () => this.off(event, fn);
    }
    /** Unsubscribe a previously registered listener. */
    off(event, fn) {
        const set = this.listeners.get(event);
        if (!set)
            return;
        set.delete(fn);
        if (set.size === 0)
            this.listeners.delete(event);
    }
    /** Subscribe for exactly one delivery, then auto-unsubscribe. */
    once(event, fn) {
        const wrapped = (payload) => {
            this.off(event, wrapped);
            fn(payload);
        };
        return this.on(event, wrapped);
    }
    /** Synchronously fan out `payload` to all listeners of `event`. */
    emit(event, payload) {
        const set = this.listeners.get(event);
        if (!set || set.size === 0)
            return;
        // Snapshot listeners so removal during iteration is safe.
        for (const fn of [...set]) {
            try {
                fn(payload);
            }
            catch {
                // Listeners must not be able to break the emit loop.
            }
        }
    }
    /** Drop every listener. Called by {@link CatetusViewer.dispose}. */
    removeAll() {
        this.listeners.clear();
    }
}
//# sourceMappingURL=events.js.map