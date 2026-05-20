// Minimal hash-based router. ~30 lines. Zero deps.
//
// Hash-based (#/about) keeps deploys host-agnostic: works on any static
// CDN without server rewrites. Switch to history mode later if needed.

export type Route = {
  pattern: RegExp;
  render: () => string | Promise<string>;
  onMount?: (root: HTMLElement) => void | Promise<void>;
};

let routes: Route[] = [];
let notFound: Route | null = null;
let rootEl: HTMLElement | null = null;

export function registerRoute(route: Route): void {
  routes.push(route);
}

export function registerNotFound(route: Route): void {
  notFound = route;
}

export function currentPath(): string {
  const h = window.location.hash.replace(/^#/, '');
  return h || '/';
}

async function render(): Promise<void> {
  if (!rootEl) return;
  const path = currentPath();
  const match = routes.find((r) => r.pattern.test(path)) ?? notFound;
  if (!match) {
    rootEl.innerHTML = '<div class="center-pane"><h2>not found</h2></div>';
    return;
  }
  const html = await match.render();
  rootEl.innerHTML = html;
  await match.onMount?.(rootEl);
  // mark active nav link
  rootEl.querySelectorAll('.nav .links a').forEach((a) => {
    const href = (a as HTMLAnchorElement).getAttribute('href') ?? '';
    if (href === '#' + path || (path === '/' && href === '#/')) {
      a.classList.add('active');
    } else {
      a.classList.remove('active');
    }
  });
  window.scrollTo(0, 0);
}

export function mountRouter(el: HTMLElement): void {
  rootEl = el;
  window.addEventListener('hashchange', () => { void render(); });
  void render();
}
