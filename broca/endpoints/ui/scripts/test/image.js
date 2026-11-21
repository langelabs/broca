import { runLoadRound } from "./load-test.js";

const inputs = [
    "A red ceramic teapot on a sunlit wooden table, still life photograph.",
    "Watercolor illustration of a lighthouse above a calm bay.",
    "A close-up photograph of a ladybug on a mint leaf.",
    "A minimalist poster of the phases of the moon, cream and navy.",
    "An old bicycle leaning against a blue-painted shopfront at dawn.",
    "An overhead photograph of sliced citrus on dark slate.",
    "A pencil sketch of a train station waiting room.",
    "A felt fox toy sitting among colorful spools of thread.",
    "A snowy courtyard seen through an arched window.",
    "A cut-paper collage of rolling hills and a winding river.",
    "A studio photograph of a clear glass filled with sparkling water.",
    "A hand-painted map of a small island with paths and a harbor.",
    "An open notebook, fountain pen, and pressed fern on a linen cloth.",
    "A rain-soaked bookshop window reflected in the pavement.",
    "A geometric mosaic of intersecting circles in teal and terracotta.",
    "A macro photograph of frost crystals on a fallen oak leaf.",
    "A quiet rooftop garden framed by neighboring brick buildings.",
    "An illustrated recipe card with small drawings of herbs and spices.",
    "A bowl of peaches under warm light in the style of an oil painting.",
    "A pastel drawing of empty rowboats tied to a wooden pier.",
];

/**
 * Time concurrent image requests using distinct inputs.
 * @param {number} round Number of concurrent requests.
 * @param {string} model Loaded model ID from the catalog.
 * @returns {Promise<number>} Round duration in milliseconds.
 */
export default function runImageLoadTest(round, model) {
    return runLoadRound("image", inputs, round, model);
}
