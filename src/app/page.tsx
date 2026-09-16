import { Tracker } from "@/components/Tracker";
import { searchGiveaways } from "@/lib/search";

export const dynamic = "force-dynamic";

export default async function Home() {
  const initial = await searchGiveaways("giveaway", "all");
  const failedHard =
    initial.results.length === 0 &&
    initial.providers.filter((provider) => provider.used).every((provider) => !provider.ok);
  return (
    <div className="flex flex-1 flex-col">
      <Tracker
        initial={initial}
        initialError={
          failedHard
            ? (initial.warnings[0] ?? "Search backends failed")
            : null
        }
      />
    </div>
  );
}
