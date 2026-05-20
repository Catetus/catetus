import { shell } from '../chrome';

export function renderNotFound(): string {
  return shell(`
    <div class="center-pane">
      <div>
        <p style="font:600 12px/1 var(--mono);color:var(--accent);letter-spacing:0.12em;text-transform:uppercase;margin:0 0 12px">404</p>
        <h1 style="font:600 28px/1.2 var(--sans);margin:0 0 12px">no such page</h1>
        <p style="color:var(--fg-dim);margin:0 0 24px">try the <a href="#/">overview</a>, the <a href="#/viewer">viewer</a>, or <a href="#/about">about</a>.</p>
      </div>
    </div>
  `);
}
