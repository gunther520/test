export default function Loading() {
  return (
    <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-4 py-8 sm:px-6">
      <div className="h-40 border-b-2 border-ink/20" />
      <div className="mt-6 grid gap-4 md:grid-cols-2" aria-busy="true">
        {Array.from({ length: 4 }).map((_, index) => (
          <div
            key={index}
            className="h-44 animate-pulse border border-ink/10 bg-ticket/80"
          />
        ))}
      </div>
    </div>
  );
}
