import { statusPayload } from "@/lib/search";

export const dynamic = "force-dynamic";

export async function GET() {
  return Response.json(statusPayload());
}
