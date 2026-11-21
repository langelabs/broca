import assert from "node:assert/strict";
import { test } from "node:test";

test("changes the loaded state of a namespaced model and returns the server state", async t => {
    const { setModelLoaded } = await import("../broca/endpoints/ui/scripts/fetch/model-api.js");
    const fetch = t.mock.method(globalThis, "fetch", async (_url, options) =>
        Response.json({ id: "example/model", loaded: JSON.parse(options.body) })
    );

    for (const loaded of [true, false]) {
        assert.deepEqual(await setModelLoaded("example/model", loaded), {
            id: "example/model",
            loaded,
        });
        const [url, options] = fetch.mock.calls.at(-1).arguments;
        assert.equal(url, "/v1/models/example%2Fmodel/loaded");
        assert.equal(options.method, "PUT");
        assert.equal(options.headers["Content-Type"], "application/json");
        assert.equal(options.body, JSON.stringify(loaded));
    }
});

test("reports the server error when a model cannot be loaded", async t => {
    const { setModelLoaded } = await import("../broca/endpoints/ui/scripts/fetch/model-api.js");
    t.mock.method(globalThis, "fetch", async () =>
        Response.json({ error: { message: "Download the model first." } }, { status: 409 })
    );

    await assert.rejects(setModelLoaded("example/model", true), /Download the model first/);
});
