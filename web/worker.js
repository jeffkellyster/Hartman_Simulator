// Runs the Python engine in Pyodide, off the page's main thread.
// Messages in:  {id, fn, args}
// Messages out: {type: "status", text} | {type: "ready", level: "basic" | "full"} | {type: "fatal", error}
//               | {type: "result", id, response: {ok, result | error}}
//
// NumPy loads first, which is enough to design and measure ("basic"). SciPy,
// which the models need, loads next in the background ("full").

const PYODIDE_VERSION = "0.27.7";
importScripts(`https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.js`);

let handle = null;

async function init() {
  postMessage({ type: "status", text: "Loading Python…" });
  const pyodide = await loadPyodide();
  postMessage({ type: "status", text: "Loading NumPy…" });
  await pyodide.loadPackage("numpy");
  postMessage({ type: "status", text: "Loading the engine…" });
  const response = await fetch("engine.zip", { cache: "no-store" });
  if (!response.ok) throw new Error(`engine.zip: HTTP ${response.status} (start the app with scripts/serve.py)`);
  pyodide.unpackArchive(await response.arrayBuffer(), "zip");
  handle = pyodide.pyimport("hartmann.app").handle;
  postMessage({ type: "ready", level: "basic" });
  await pyodide.loadPackage("scipy");
  postMessage({ type: "ready", level: "full" });
}

const ready = init().catch((err) => {
  postMessage({ type: "fatal", error: String(err) });
  throw err;
});

onmessage = async (event) => {
  const { id, fn, args } = event.data;
  try {
    while (handle === null) await Promise.race([ready, new Promise((r) => setTimeout(r, 50))]);
    const response = JSON.parse(handle(JSON.stringify({ fn, args })));
    postMessage({ type: "result", id, response });
  } catch (err) {
    postMessage({ type: "result", id, response: { ok: false, error: String(err) } });
  }
};
