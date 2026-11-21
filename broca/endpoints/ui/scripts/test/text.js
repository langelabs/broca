import { runLoadRound } from "./load-test.js";

const inputs = [
    "Summarize why public libraries matter in two sentences.",
    "Give three names for a neighborhood bakery.",
    "Translate 'The museum opens tomorrow' into German.",
    "Explain the difference between a comet and an asteroid in one sentence.",
    "Sort these words alphabetically: pear, fig, banana, date.",
    "Write a polite two-sentence email to reschedule a dentist appointment.",
    "Return a JSON object with keys city and country for Kyoto.",
    "Write a three-line poem about fog lifting from a river.",
    "Suggest two vegetarian fillings for a picnic sandwich.",
    "Convert 150 minutes to hours and minutes.",
    "Rewrite 'The parcel was delivered by Lee' in active voice.",
    "Explain how a bicycle bell works for a curious child.",
    "Give directions from a library to a park one block south and two blocks east.",
    "Find the adjective in the sentence 'The quiet train arrived early.'",
    "Write a one-sentence caption for a photo of a cat asleep on a book.",
    "Create a two-step checklist for watering houseplants.",
    "Estimate how many apples remain after sharing 12 equally among three people.",
    "Describe what happens when warm air meets a cold window.",
    "Write a short invitation to a Saturday board-game night.",
    "Give one example each of a simile and a metaphor about rain.",
];

/**
 * Time concurrent text requests using distinct inputs.
 * @param {number} round Number of concurrent requests.
 * @param {string} model Loaded model ID from the catalog.
 * @returns {Promise<number>} Round duration in milliseconds.
 */
export default function runTextLoadTest(round, model) {
    return runLoadRound("text", inputs, round, model);
}
