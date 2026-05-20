import { mountRouter, registerRoute, registerNotFound } from './router';
import { renderHome } from './pages/home';
import { renderViewer } from './pages/viewer';
import { renderAbout } from './pages/about';
import { renderNotFound } from './pages/notFound';

registerRoute({ pattern: /^\/$/, render: renderHome });
registerRoute({ pattern: /^\/viewer\/?$/, render: renderViewer });
registerRoute({ pattern: /^\/about\/?$/, render: renderAbout });
registerNotFound({ pattern: /.*/, render: renderNotFound });

const root = document.getElementById('app');
if (root) mountRouter(root);
