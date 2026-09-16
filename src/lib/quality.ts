export function hostnameOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./i, "").toLowerCase();
  } catch {
    return "";
  }
}

export function displayHost(url: string): string {
  return hostnameOf(url) || "unknown host";
}

function pathOf(url: string): string {
  try {
    const parsed = new URL(url);
    return `${parsed.pathname}${parsed.search}`.toLowerCase();
  } catch {
    return "";
  }
}

const CONTEST_HOSTS = [
  "gleam.io",
  "woobox.com",
  "kingsumo.com",
  "rafflecopter.com",
  "viralsweep.com",
  "rafflebox.ca",
  "shortstack.com",
  "contestfactory.com",
];

const JUNK_HOSTS = [
  "cnn.com",
  "bbc.com",
  "bbc.co.uk",
  "yahoo.com",
  "wikipedia.org",
  "wikihow.com",
  "nytimes.com",
  "theguardian.com",
  "reuters.com",
  "bloomberg.com",
  "forbes.com",
  "businessinsider.com",
  "insider.com",
  "buzzfeed.com",
  "huffpost.com",
  "huffingtonpost.com",
  "dailymail.co.uk",
  "nypost.com",
  "foxnews.com",
  "msn.com",
  "news.google.com",
  "apnews.com",
  "npr.org",
  "usatoday.com",
  "latimes.com",
  "washingtonpost.com",
  "wsj.com",
  "time.com",
  "newsweek.com",
  "people.com",
  "today.com",
  "cnet.com",
  "theverge.com",
  "techcrunch.com",
  "mashable.com",
  "lifehacker.com",
  "wired.com",
  "independent.co.uk",
  "telegraph.co.uk",
  "scmp.com",
  "hk01.com",
  "mingpao.com",
  "rthk.hk",
  "medium.com",
];

function hostMatches(host: string, list: string[]): boolean {
  return list.some((known) => host === known || host.endsWith(`.${known}`));
}

export function isContestHost(url: string): boolean {
  return hostMatches(hostnameOf(url), CONTEST_HOSTS);
}

export function isJunkNewsHost(url: string): boolean {
  if (isContestHost(url)) return false;
  return hostMatches(hostnameOf(url), JUNK_HOSTS);
}

export function isEnterablePostUrl(url: string): boolean {
  if (isContestHost(url)) return true;
  const host = hostnameOf(url);
  const path = pathOf(url);

  if (host === "instagram.com" || host.endsWith(".instagram.com") || host === "instagr.am") {
    return /\/(p|reel|reels|stories|tv)\//.test(path);
  }
  if (host === "facebook.com" || host.endsWith(".facebook.com") || host === "fb.com" || host === "fb.watch") {
    return (
      /\/(posts|photos|videos|reel|reels|watch|stories|share|permalink\.php|story\.php|photo\.php)\b/.test(
        path,
      ) || /story_fbid=|\/groups\/[^/]+\/(posts|permalink)/.test(path)
    );
  }
  if (
    host === "x.com" ||
    host === "twitter.com" ||
    host.endsWith(".x.com") ||
    host.endsWith(".twitter.com")
  ) {
    return /\/status\/\d+/.test(path);
  }
  if (host === "youtube.com" || host.endsWith(".youtube.com")) {
    return /\/watch\b|\/shorts\/|\/live\//.test(path);
  }
  if (host === "youtu.be") return path.length > 1;
  if (host === "twitch.tv" || host.endsWith(".twitch.tv")) {
    if (host === "clips.twitch.tv") return true;
    return /\/videos\/|\/clip\//.test(path);
  }
  if (host === "tiktok.com" || host.endsWith(".tiktok.com")) {
    if (host.startsWith("vm.")) return true;
    return /\/video\/|\/t\//.test(path);
  }
  if (host === "reddit.com" || host.endsWith(".reddit.com")) {
    return /\/comments\//.test(path);
  }
  return false;
}

const CONTEST_INTENT =
  /comment to win|tag (a friend|friends)|follow to (enter|win)|retweet to win|\brt to win\b|enter to win|giveaway poll|poll to win|lucky draw|sweepstake|raffle|(ends?|ending|closes?)\s+(on|at|this|by|tomorrow|tonight|soon|in)\b|\bdeadline\b|抽獎|免費送|送你|送出|有獎|留言.{0,6}(贏|抽)|標註|參加抽/;

const WEAK_GIVEAWAY = /\bgiveaway\b|give away|抽獎|\bcontest\b|\bprize\b/;

const NON_ENTERABLE =
  /congratulat\w+\s+to\s+(the\s+|our\s+|those\s+|all\s+)?(winner|those who)|winners?\s+(announced|chosen|selected|picked|are)|how to\s+(run|host|start|do|create)\s+a?\s*(giveaway|contest|raffle)|(best|top)\s+\d+\s+giveaways|giveaway\s+(ideas|tips|guide|recap|round-?up|results|winners)|stimulus giveaway|tax giveaway|giveaway to (israel|ukraine)/i;

export function hasContestIntent(title: string, snippet = ""): boolean {
  const text = `${title} ${snippet}`;
  return CONTEST_INTENT.test(text.toLowerCase()) || /抽獎|免費送|送你|送出/.test(text);
}

export function isNonEnterableCopy(title: string, snippet = ""): boolean {
  return NON_ENTERABLE.test(`${title} ${snippet}`);
}

export function mentionsGiveaway(title: string, snippet = ""): boolean {
  const text = `${title} ${snippet}`;
  return WEAK_GIVEAWAY.test(text.toLowerCase()) || /抽獎|免費送/.test(text);
}

/** Keep enterable posts/hosts, or copy with contest-intent. Drop news/how-tos. */
export function isEnterableGiveaway(item: {
  url: string;
  title: string;
  snippet: string;
  endsAt?: string;
}): boolean {
  if (isJunkNewsHost(item.url) && !isContestHost(item.url)) return false;
  if (isNonEnterableCopy(item.title, item.snippet) && !hasContestIntent(item.title, item.snippet)) {
    return false;
  }
  const contesty =
    hasContestIntent(item.title, item.snippet) ||
    mentionsGiveaway(item.title, item.snippet) ||
    Boolean(item.endsAt) ||
    isContestHost(item.url);
  if (!contesty) return false;
  if (isEnterablePostUrl(item.url) || isContestHost(item.url)) return true;
  if (hasContestIntent(item.title, item.snippet) || item.endsAt) return true;
  return false;
}

export function qualityScore(item: {
  url: string;
  title: string;
  snippet: string;
  endsAt?: string;
}): number {
  if (!isEnterableGiveaway(item)) return 0;
  let score = 1;
  if (isEnterablePostUrl(item.url) || isContestHost(item.url)) score += 8;
  if (hasContestIntent(item.title, item.snippet)) score += 5;
  if (item.endsAt) score += 2;
  if (isNonEnterableCopy(item.title, item.snippet)) score -= 4;
  return score;
}
