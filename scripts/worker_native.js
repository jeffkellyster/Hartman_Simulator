// Development stand-in for web/worker.js, served by `scripts/serve.py --native`.
// Same messages, but each call goes to the engine running natively at /api
// instead of to Pyodide, so no CDN is needed.

postMessage({ type: "status", text: "Native engine (development mode)" });
postMessage({ type: "ready", level: "basic" });
postMessage({ type: "ready", level: "full" });

onmessage = async (event) => {
  const { id, fn, args } = event.data;
  try {
    const response = await fetch("/api", { method: "POST", body: JSON.stringify({ fn, args }) });
    postMessage({ type: "result", id, response: await response.json() });
  } catch (err) {
    postMessage({ type: "result", id, response: { ok: false, error: String(err) } });
  }
};
