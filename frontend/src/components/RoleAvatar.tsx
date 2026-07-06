import {
  BadgeAlert,
  Bot,
  CircleQuestionMark,
  Crosshair,
  Eye,
  FlaskConical,
  HeartPulse,
  Shield,
  UserRound,
} from "lucide-react";
import { Avatar, AvatarBadge, AvatarFallback } from "@/components/ui/avatar";
import { cn } from "@/lib/utils";

type RoleAvatarSize = "sm" | "default" | "lg";

type RoleAvatarProps = {
  role?: string | null;
  team?: string | null;
  seat?: number | string | null;
  alive?: boolean;
  reveal?: boolean;
  size?: RoleAvatarSize;
  className?: string;
  badgeClassName?: string;
};

const ROLE_LABEL: Record<string, string> = {
  werewolf: "狼人",
  seer: "预言家",
  witch: "女巫",
  guard: "守卫",
  hunter: "猎人",
  doctor: "医生",
  villager: "村民",
};

function roleTone(role?: string | null, team?: string | null, reveal = true): string {
  if (!reveal) return "hidden";
  if (role === "werewolf" || team === "werewolves" || team === "wolf") return "werewolf";
  if (role) return role;
  if (team === "village" || team === "good") return "villager";
  return "hidden";
}

function RoleGlyph({ role, tone, alive }: { role?: string | null; tone: string; alive: boolean }) {
  const props = { "aria-hidden": true, className: "size-4" };
  if (!alive) return <BadgeAlert {...props} />;
  if (role === "seer") return <Eye {...props} />;
  if (role === "witch") return <FlaskConical {...props} />;
  if (role === "guard") return <Shield {...props} />;
  if (role === "hunter") return <Crosshair {...props} />;
  if (role === "doctor") return <HeartPulse {...props} />;
  if (tone === "werewolf") return <BadgeAlert {...props} />;
  if (tone === "villager") return <Bot {...props} />;
  if (tone === "hidden") return <CircleQuestionMark {...props} />;
  return <UserRound {...props} />;
}

export function RoleAvatar({
  role,
  team,
  seat,
  alive = true,
  reveal = true,
  size = "default",
  className,
  badgeClassName,
}: RoleAvatarProps) {
  const tone = roleTone(role, team, reveal);
  const title = reveal && role ? `${seat ?? "?"}号 · ${roleLabel(role)}` : `${seat ?? "?"}号 · 身份隐藏`;
  const toneClass =
    tone === "werewolf" ? "bg-red-950 text-red-100 ring-red-500/35" :
    tone === "seer" ? "bg-sky-950 text-sky-100 ring-sky-500/35" :
    tone === "witch" ? "bg-violet-950 text-violet-100 ring-violet-500/35" :
    tone === "guard" ? "bg-emerald-950 text-emerald-100 ring-emerald-500/35" :
    tone === "hunter" ? "bg-amber-950 text-amber-100 ring-amber-500/35" :
    tone === "doctor" ? "bg-teal-950 text-teal-100 ring-teal-500/35" :
    tone === "villager" ? "bg-zinc-800 text-zinc-100 ring-zinc-500/35" :
    "bg-muted text-muted-foreground ring-border";

  return (
    <Avatar
      size={size}
      className={cn("ring-2", toneClass, !alive && "opacity-55 grayscale", className)}
      title={title}
      aria-label={title}
    >
      <AvatarFallback className={cn("flex flex-col gap-0.5 bg-transparent text-current", toneClass)}>
        <RoleGlyph role={reveal ? role : undefined} tone={tone} alive={alive} />
        <span className="text-[10px] font-semibold leading-none">{seat ?? "?"}</span>
      </AvatarFallback>
      {alive && <AvatarBadge className={cn("bg-emerald-400 ring-background", badgeClassName)} />}
    </Avatar>
  );
}

export function roleLabel(role?: string | null): string {
  return role ? ROLE_LABEL[role] || role : "未知";
}
