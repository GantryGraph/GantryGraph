"""Shared Playwright stealth utilities — used by both actions.BrowserTools and perception.WebPage.

Applies layered patches to reduce bot-detection signals:
  - JS init script applied to every new page context
  - Realistic context kwargs (user-agent, viewport, locale, timezone)
  - Chromium launch arg that disables the automation flag

Optional deeper coverage: ``pip install playwright-stealth``.
When installed, :func:`apply_to_page` also runs its patches on top of ours.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Page

try:
    from playwright_stealth import stealth_async as _stealth_async

    _HAS_PLAYWRIGHT_STEALTH = True
except ImportError:
    _HAS_PLAYWRIGHT_STEALTH = False

VIEWPORT_W = 1366
VIEWPORT_H = 768

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]

# All values must stay consistent with VIEWPORT_W/H above.
_STEALTH_JS = """
(() => {
    // 1. Remove the webdriver flag — caught by every basic bot detector
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

    // 2. Chrome runtime object (absent in vanilla headless builds)
    if (!window.chrome) {
        window.chrome = {
            runtime: {},
            loadTimes: function(){},
            csi: function(){},
            app: {},
        };
    }

    // 3. Plugins list (empty in headless Chrome)
    if (navigator.plugins.length === 0) {
        Object.defineProperty(navigator, 'plugins', {
            get: () => [
                'PDF Viewer', 'Chrome PDF Viewer', 'Chromium PDF Viewer',
                'Microsoft Edge PDF Viewer', 'WebKit built-in PDF',
            ].map(n => ({ name: n })),
        });
    }

    // 4. Languages
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });

    // 5. Notifications permission — headless returns 'denied', real browsers return 'default'
    if (window.navigator.permissions) {
        const _q = window.navigator.permissions.query.bind(window.navigator.permissions);
        window.navigator.permissions.query = p =>
            p.name === 'notifications'
                ? Promise.resolve({ state: 'default', onchange: null })
                : _q(p);
    }

    // 6. Hardware concurrency and device memory (headless often reports low values)
    Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
    try { Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 }); } catch(e) {}

    // 7. Network connection info
    if (!navigator.connection) {
        Object.defineProperty(navigator, 'connection', {
            get: () => ({ effectiveType: '4g', rtt: 50, downlink: 10, saveData: false }),
        });
    }

    // 8. Screen and window geometry — must match the Playwright viewport
    Object.defineProperty(screen, 'width',       { get: () => 1366 });
    Object.defineProperty(screen, 'height',      { get: () => 768  });
    Object.defineProperty(screen, 'availWidth',  { get: () => 1366 });
    Object.defineProperty(screen, 'availHeight', { get: () => 728  });
    Object.defineProperty(screen, 'colorDepth',  { get: () => 24   });
    Object.defineProperty(screen, 'pixelDepth',  { get: () => 24   });
    Object.defineProperty(window, 'outerWidth',  { get: () => 1366 });
    Object.defineProperty(window, 'outerHeight', { get: () => 768  });

    // 9. WebGL vendor / renderer — a major canvas-style fingerprinting vector
    try {
        const _spoof = (ctx) => {
            const _orig = ctx.getParameter.bind(ctx);
            WebGLRenderingContext.prototype.getParameter = function(p) {
                if (p === 37445) return 'Intel Inc.';              // UNMASKED_VENDOR_WEBGL
                if (p === 37446) return 'Intel Iris OpenGL Engine'; // UNMASKED_RENDERER_WEBGL
                return _orig(p);
            };
        };
        _spoof(WebGLRenderingContext.prototype);
        if (typeof WebGL2RenderingContext !== 'undefined') {
            _spoof(WebGL2RenderingContext.prototype);
        }
    } catch(e) {}

    // 10. maxTouchPoints — desktop = 0; headless sometimes leaks a non-zero value
    try { Object.defineProperty(navigator, 'maxTouchPoints', { get: () => 0 }); } catch(e) {}
})();
"""


def context_kwargs() -> dict[str, Any]:
    """Return keyword arguments for ``browser.new_context()`` or ``launch_persistent_context()``."""
    return {
        "user_agent": _USER_AGENT,
        "viewport": {"width": VIEWPORT_W, "height": VIEWPORT_H},
        "locale": "en-US",
        "timezone_id": "America/New_York",
        "extra_http_headers": {"Accept-Language": "en-US,en;q=0.9"},
    }


async def apply_to_context(context: BrowserContext) -> None:
    """Add the stealth init script to *context* so every new page gets it."""
    await context.add_init_script(_STEALTH_JS)


async def apply_to_page(page: Page) -> None:
    """Run ``playwright-stealth`` on *page* if the package is installed."""
    if _HAS_PLAYWRIGHT_STEALTH:
        await _stealth_async(page)
