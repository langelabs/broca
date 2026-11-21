import assert from "node:assert/strict";
import { test } from "node:test";

import * as runner from "../broca/endpoints/ui/scripts/test/load-test.js";

const modalities = ["audio", "decision", "embedding", "image", "text", "video"];

for (const modality of modalities) {
    test(`${modality} sends the requested number of distinct concurrent inputs`, async t => {
        const requests = [];
        let gate;
        t.mock.method(globalThis, "fetch", async (url, options) => {
            requests.push({ url, method: options.method, ...JSON.parse(options.body) });
            await gate;
            return new Response("result");
        });

        const { default: run } = await import(`../broca/endpoints/ui/scripts/test/${modality}.js`)
            .catch(error => {
                assert.fail(`Expected a browser load test for ${modality}: ${error.message}`);
            });
        for (let round = 1; round <= 20; round++) {
            let release;
            gate = new Promise(resolve => { release = resolve; });
            const result = run(round, "example/model");
            try {
                assert.equal(requests.length, round);
                assert.equal(new Set(requests.map(request => request.input)).size, round);
                for (const { url, method, model, input } of requests) {
                    assert.equal(url, `/${modality}`);
                    assert.equal(method, "POST");
                    assert.equal(model, "example/model");
                    assert.ok(input.trim());
                    if (modality === "decision") {
                        const { state, decision } = JSON.parse(input);
                        assert.ok(state);
                        assert.equal(decision.kind, "noul");
                        assert.ok(decision.instructions && decision.yes_description && decision.no_description);
                    }
                }
            } finally {
                release();
                const elapsed = await result;
                assert.ok(Number.isFinite(elapsed) && elapsed >= 0);
            }
            requests.length = 0;
        }
    });
}

test("a failed request reports the error after the other response body finishes", async t => {
    const { runLoadRound } = runner;
    assert.equal(typeof runLoadRound, "function");

    let finishDownload;
    let readingBody;
    const startedDownload = new Promise(resolve => { readingBody = resolve; });
    const stream = new ReadableStream({
        start(controller) { finishDownload = () => controller.close(); },
        pull() { readingBody(); },
    }, { highWaterMark: 0 });
    t.mock.method(globalThis, "fetch", async (_url, options) => {
        if (JSON.parse(options.body).input === "fail") {
            return Response.json({ detail: "The model was unloaded." }, { status: 409 });
        }
        return new Response(stream);
    });

    let finished = false;
    const result = runLoadRound("video", ["fail", "download"], 2, "example/model");
    const failure = assert.rejects(result, /1 of 2 requests failed.*The model was unloaded/);
    result.catch(() => { finished = true; });
    try {
        await Promise.race([startedDownload, failure]);
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(finished, false);
    } finally {
        finishDownload();
        await failure;
    }
});

test("invalid rounds are rejected before sending any requests", async t => {
    const { runLoadRound } = runner;
    assert.equal(typeof runLoadRound, "function");
    const fetch = t.mock.method(globalThis, "fetch", async () => new Response());
    for (const round of [0, 1.5, 21]) {
        await assert.rejects(runLoadRound("text", Array(20).fill("prompt"), round, "example/model"), RangeError);
    }
    assert.equal(fetch.mock.callCount(), 0);
});
