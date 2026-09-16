import {
  displayHost,
  isEnterableGiveaway,
  isEnterablePostUrl,
  isJunkNewsHost,
  qualityScore,
} from "../src/lib/quality";

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

assert(isEnterablePostUrl("https://www.instagram.com/p/AbC123/"), "ig p");
assert(isEnterablePostUrl("https://www.instagram.com/reel/AbC123/"), "ig reel");
assert(isEnterablePostUrl("https://www.facebook.com/brand/posts/123"), "fb posts");
assert(isEnterablePostUrl("https://www.facebook.com/permalink.php?story_fbid=1"), "fb permalink");
assert(isEnterablePostUrl("https://x.com/user/status/123456"), "x status");
assert(isEnterablePostUrl("https://www.youtube.com/watch?v=abc"), "yt watch");
assert(isEnterablePostUrl("https://www.tiktok.com/@u/video/123"), "tt video");
assert(isEnterablePostUrl("https://www.reddit.com/r/giveaways/comments/abc/hi/"), "reddit comments");
assert(isEnterablePostUrl("https://gleam.io/xyz/win"), "gleam");
assert(!isEnterablePostUrl("https://www.instagram.com/brand/"), "ig profile");
assert(!isEnterablePostUrl("https://www.facebook.com/brand/"), "fb page");
assert(!isEnterablePostUrl("https://x.com/user"), "x profile");

assert(isJunkNewsHost("https://www.cnn.com/2026/09/01/giveaway"), "cnn");
assert(isJunkNewsHost("https://en.wikipedia.org/wiki/Giveaway"), "wiki");
assert(isJunkNewsHost("https://news.google.com/rss/articles/abc"), "gnews");
assert(
  !isJunkNewsHost("https://news.google.com/rss/articles/abc", "instagram.com"),
  "gnews + ig source not junk",
);
assert(!isJunkNewsHost("https://www.instagram.com/p/abc"), "ig not junk");

const ig = {
  url: "https://www.instagram.com/p/abc/",
  title: "AirPods giveaway",
  snippet: "posted today",
};
assert(isEnterableGiveaway(ig), "ig giveaway post");
assert(qualityScore(ig) >= 8, "ig scores high");

assert(
  !isEnterableGiveaway({
    url: "https://www.facebook.com/SomeBrand/",
    title: "Giveaway",
    snippet: "Check our page",
  }),
  "fb page lone giveaway dropped",
);

assert(
  isEnterableGiveaway({
    url: "https://www.facebook.com/SomeBrand/posts/99",
    title: "Giveaway",
    snippet: "Check our page",
  }),
  "fb post lone giveaway kept",
);

assert(
  !isEnterableGiveaway({
    url: "https://www.cnn.com/world/giveaway-story",
    title: "Comment to win a car giveaway",
    snippet: "ends tomorrow",
  }),
  "cnn dropped even with intent",
);

assert(
  isEnterableGiveaway({
    url: "https://brand.example.com/win",
    title: "Comment to win our raffle",
    snippet: "Tag a friend. Deadline 20 Sep 2026",
  }),
  "brand site with intent kept",
);

assert(
  !isEnterableGiveaway({
    url: "https://blog.example.com/best-giveaways",
    title: "Giveaway",
    snippet: "A list of giveaways this week",
  }),
  "lone giveaway on blog dropped",
);

assert(
  !isEnterableGiveaway({
    url: "https://www.facebook.com/brand/posts/1",
    title: "Congratulations to those who got the giveaway",
    snippet: "Winners announced yesterday",
  }),
  "winner recap dropped",
);

assert(
  isEnterableGiveaway({
    url: "https://www.instagram.com/reel/xyz/",
    title: "抽獎送你 iPhone",
    snippet: "免費送",
  }),
  "HK Chinese contest kept",
);

assert(
  !isEnterableGiveaway({
    url: "https://www.bbc.com/news/giveaway",
    title: "Lucky draw coverage",
    snippet: "",
  }),
  "bbc dropped",
);

assert(displayHost("https://www.instagram.com/p/abc/") === "instagram.com", "display host");

assert(
  !isEnterableGiveaway({
    url: "https://www.facebook.com/",
    title: "Comment on this post and tag a friend giveaway",
    snippet: "Must be following me",
  }),
  "bare facebook home dropped",
);

assert(
  isEnterableGiveaway({
    url: "https://news.google.com/rss/articles/CBMiabc",
    sourceHost: "facebook.com",
    title: "Quick giveaway — comment on this post and tag a friend",
    snippet: "Must be following me",
  }),
  "google news facebook source with intent kept",
);

assert(
  !isEnterableGiveaway({
    url: "https://news.google.com/rss/articles/CBMiabc",
    sourceHost: "facebook.com",
    title: "Congratulations to our Fifth Week Lucky Draw Winners!",
    snippet: "The Onam celebrations",
  }),
  "lucky draw winners recap dropped",
);

assert(
  !isEnterableGiveaway({
    url: "https://news.google.com/rss/articles/CBMiabc",
    sourceHost: "cnn.com",
    title: "Comment to win a car giveaway",
    snippet: "ends tomorrow",
  }),
  "google news cnn source dropped",
);

console.log("ok quality filter");
