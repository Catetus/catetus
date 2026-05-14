import { describe, expect, it, vi } from 'vitest';
import { Emitter } from '../events.js';

type Map = {
  tick: { n: number };
  done: { ok: boolean };
};

describe('Emitter', () => {
  it('delivers payloads to subscribed listeners', () => {
    const e = new Emitter<Map>();
    const fn = vi.fn();
    e.on('tick', fn);
    e.emit('tick', { n: 1 });
    e.emit('tick', { n: 2 });
    expect(fn).toHaveBeenCalledTimes(2);
    expect(fn).toHaveBeenNthCalledWith(1, { n: 1 });
    expect(fn).toHaveBeenNthCalledWith(2, { n: 2 });
  });

  it('off() unsubscribes a listener', () => {
    const e = new Emitter<Map>();
    const fn = vi.fn();
    e.on('tick', fn);
    e.emit('tick', { n: 1 });
    e.off('tick', fn);
    e.emit('tick', { n: 2 });
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it('on() returns an unsubscribe handle', () => {
    const e = new Emitter<Map>();
    const fn = vi.fn();
    const unsub = e.on('tick', fn);
    unsub();
    e.emit('tick', { n: 99 });
    expect(fn).not.toHaveBeenCalled();
  });

  it('once() fires exactly one time', () => {
    const e = new Emitter<Map>();
    const fn = vi.fn();
    e.once('done', fn);
    e.emit('done', { ok: true });
    e.emit('done', { ok: false });
    expect(fn).toHaveBeenCalledTimes(1);
    expect(fn).toHaveBeenCalledWith({ ok: true });
  });

  it('a listener throwing does not break the emit loop', () => {
    const e = new Emitter<Map>();
    const good = vi.fn();
    e.on('tick', () => {
      throw new Error('boom');
    });
    e.on('tick', good);
    e.emit('tick', { n: 7 });
    expect(good).toHaveBeenCalledWith({ n: 7 });
  });

  it('removeAll() clears every subscription', () => {
    const e = new Emitter<Map>();
    const a = vi.fn();
    const b = vi.fn();
    e.on('tick', a);
    e.on('done', b);
    e.removeAll();
    e.emit('tick', { n: 1 });
    e.emit('done', { ok: true });
    expect(a).not.toHaveBeenCalled();
    expect(b).not.toHaveBeenCalled();
  });
});
