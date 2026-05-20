// Page chrome: nav + footer. Wraps each route body.

export function shell(body: string, opts: { wide?: boolean } = {}): string {
  return `
    <div class="shell">
      ${nav()}
      <main>
        <div class="container${opts.wide ? ' wide' : ''}">
          ${body}
        </div>
      </main>
      ${footer()}
    </div>
  `;
}

function nav(): string {
  return `
    <nav class="nav">
      <a class="brand" href="#/"><span>SPLAT</span><span class="dot">·</span><span>FORGE</span></a>
      <div class="links">
        <a href="#/">overview</a>
        <a href="#/viewer">viewer</a>
        <a href="#/about">about</a>
        <a href="https://github.com/Catetus/catetus" rel="noopener" target="_blank">github</a>
      </div>
    </nav>
  `;
}

function footer(): string {
  const year = new Date().getFullYear();
  return `
    <footer class="footer">
      <div>
        Catetus &middot; ${year} &middot;
        <a href="#/about">methodology</a> &middot;
        72-view orbit (24 az × 3 el), 512², gsplat sh=3
      </div>
      <div>research preview &middot; no warranties &middot; apache-2.0</div>
    </footer>
  `;
}
