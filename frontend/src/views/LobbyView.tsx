import { useEffect, useState } from "react";
import { ArrowRight, Bot, KeyRound, RadioTower, Settings2, UserRound, Users } from "lucide-react";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { createRoom, getConfig, getProviders } from "../lib/api";
import type { RoomAuth } from "../lib/api";
import type { ProviderMeta } from "../lib/types";

const DEFAULT_NAMES = ["阿白", "林深", "苏离", "陈默", "夏野", "周遥", "顾川", "沈砚", "叶舟", "韩青", "南星", "陆迟"];

export function LobbyView({ onCreated }: { onCreated: (roomId: string, auth: RoomAuth) => void }) {
  const [names, setNames] = useState<string[]>(DEFAULT_NAMES.slice(0, 8));
  const [providers, setProviders] = useState<Record<string, ProviderMeta>>({});
  const [defaultCfg, setDefaultCfg] = useState<any>({});
  const [cfg, setCfg] = useState<any>({
    provider: "openai",
    model: "",
    api_base: "",
    api_key: "",
    temperature: 0.85,
    max_tokens: 0,
    use_json_format: false,
  });
  const [humanSeats, setHumanSeats] = useState<number[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    getProviders().then(setProviders).catch(() => {});
    getConfig()
      .then((config) => {
        setDefaultCfg(config);
        setCfg((prev: any) => ({
          ...prev,
          provider: config.provider || prev.provider || "openai",
          model: config.model || "",
          api_base: "",
          temperature: config.temperature ?? 0.85,
          max_tokens: config.max_tokens ?? 0,
          use_json_format: config.use_json_format ?? false,
        }));
      })
      .catch(() => {});
  }, []);

  const providerEntries: [string, ProviderMeta][] = Object.entries(providers).length
    ? Object.entries(providers)
    : [["openai", { label: "OpenAI-compatible", hint: "", default_api_base: "", default_model: "" }]];
  const selectedProvider = providerEntries.find(([key]) => key === cfg.provider)?.[1];

  const setCount = (count: number) => {
    setNames(DEFAULT_NAMES.slice(0, count));
    setHumanSeats((prev) => prev.filter((seat) => seat <= count));
  };

  const toggleHuman = (seat: number) => {
    setHumanSeats((prev) => (prev.includes(seat) ? prev.filter((candidate) => candidate !== seat) : [...prev, seat].sort((a, b) => a - b)));
  };

  const submit = async () => {
    setBusy(true);
    setErr("");
    try {
      const modelConfig: Record<string, any> = {};
      for (const [key, value] of Object.entries(cfg)) {
        if (value !== "" && value !== null && value !== undefined) modelConfig[key] = value;
      }
      const res = await createRoom({ player_names: names, model_config: modelConfig, human_seats: humanSeats });
      onCreated(res.room_id, { admin_token: res.admin_token, seat_tokens: res.seat_tokens || {} });
    } catch (error: any) {
      setErr(String(error.message || error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid min-h-[calc(100svh-72px)] gap-4 lg:grid-cols-[minmax(0,1fr)_380px] xl:grid-cols-[minmax(0,1fr)_420px]">
      <Card className="min-h-0 bg-card/95 shadow-sm">
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <CardTitle className="flex items-center gap-2 text-xl">
                <Users className="size-5" />
                创建真实对局
              </CardTitle>
              <CardDescription className="mt-1">
                选择人数、昵称和真人座位，创建后进入等待室启动真实模型对局。
              </CardDescription>
            </div>
            <div className="flex flex-wrap gap-2">
              <Badge variant="outline" className="gap-1.5">
                <RadioTower className="size-3" />
                {selectedProvider?.label || cfg.provider}
              </Badge>
              <Badge variant={defaultCfg.api_key_configured ? "default" : "destructive"} className="gap-1.5">
                <KeyRound className="size-3" />
                {defaultCfg.api_key_configured ? "Key 已配置" : "需要 Key"}
              </Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent className="min-h-0 space-y-5">
          <section className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <Label>玩家人数</Label>
                <p className="mt-1 text-sm text-muted-foreground">座位号固定，方便对局中记忆和投票。</p>
              </div>
              <div className="flex flex-wrap gap-2">
                {[6, 7, 8, 9, 10, 11, 12].map((count) => (
                  <Button key={count} type="button" size="sm" variant={names.length === count ? "default" : "outline"} onClick={() => setCount(count)}>
                    {count}人
                  </Button>
                ))}
              </div>
            </div>
          </section>

          <Separator />

          <section className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <Label>座位与昵称</Label>
                <p className="mt-1 text-sm text-muted-foreground">点击图标切换 AI / 真人座位。</p>
              </div>
              <Badge variant="outline">真人 {humanSeats.length} 席</Badge>
            </div>
            <ScrollArea className="h-[clamp(220px,calc(100svh-360px),560px)] rounded-lg border bg-background/45">
              <div className="grid gap-2 p-3 sm:grid-cols-2 2xl:grid-cols-3">
                {names.map((name, index) => {
                  const seat = index + 1;
                  const human = humanSeats.includes(seat);
                  return (
                    <Card key={seat} size="sm" className="bg-card/90 shadow-none">
                      <CardContent className="flex items-center gap-2 px-3">
                        <Badge variant="outline" className="w-12 justify-center">{seat}号</Badge>
                        <Input
                          value={name}
                          onChange={(event) => setNames((prev) => prev.map((value, i) => (i === index ? event.target.value : value)))}
                          aria-label={`${seat}号昵称`}
                        />
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <Button
                              type="button"
                              size="icon-sm"
                              variant={human ? "default" : "outline"}
                              onClick={() => toggleHuman(seat)}
                              aria-label={human ? "真人座位" : "AI 座位"}
                            >
                              {human ? <UserRound className="size-4" /> : <Bot className="size-4" />}
                            </Button>
                          </TooltipTrigger>
                          <TooltipContent>{human ? "真人座位" : "AI 座位"}</TooltipContent>
                        </Tooltip>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </ScrollArea>
          </section>
        </CardContent>
      </Card>

      <aside className="min-h-0 space-y-4">
        <Card className="flex min-h-[360px] bg-card/95 shadow-sm lg:h-[calc(100svh-72px)] lg:max-h-[calc(100svh-72px)]">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings2 className="size-4" />
              开局设置
            </CardTitle>
            <CardDescription>模型调用、房间和 WebSocket 都使用真实后端。</CardDescription>
          </CardHeader>
          <CardContent className="min-h-0 flex-1 overflow-hidden">
            <ScrollArea className="h-full pr-3">
              <div className="space-y-4">
                <div className="grid gap-2">
                  <InfoLine label="人数" value={`${names.length} 人`} />
                  <InfoLine label="真人座位" value={humanSeats.length ? humanSeats.map((seat) => `${seat}号`).join(" / ") : "无"} />
                  <InfoLine label="默认模型" value={cfg.model || defaultCfg.model || "后端默认"} />
                </div>

                <Accordion type="single" collapsible>
                  <AccordionItem value="model">
                    <AccordionTrigger>高级模型设置</AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-4 pt-2">
                        <div className="space-y-2">
                          <Label htmlFor="default-provider">Provider</Label>
                          <Select value={cfg.provider} onValueChange={(provider) => setCfg({ ...cfg, provider })}>
                            <SelectTrigger id="default-provider" className="w-full">
                              <SelectValue placeholder="选择 Provider" />
                            </SelectTrigger>
                            <SelectContent>
                              {providerEntries.map(([key, value]) => (
                                <SelectItem key={key} value={key}>{value.label || key}</SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="default-model">Model</Label>
                          <Input
                            id="default-model"
                            value={cfg.model}
                            placeholder={defaultCfg.model || selectedProvider?.default_model || "模型 ID"}
                            onChange={(event) => setCfg({ ...cfg, model: event.target.value })}
                          />
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="default-api-base">API Base</Label>
                          <Input
                            id="default-api-base"
                            value={cfg.api_base}
                            placeholder={defaultCfg.api_base || selectedProvider?.default_api_base || "后端默认"}
                            onChange={(event) => setCfg({ ...cfg, api_base: event.target.value })}
                          />
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="default-api-key">API Key</Label>
                          <Input
                            id="default-api-key"
                            type="password"
                            value={cfg.api_key}
                            placeholder={defaultCfg.api_key_configured ? "已配置，留空沿用" : "填写 key"}
                            onChange={(event) => setCfg({ ...cfg, api_key: event.target.value })}
                          />
                        </div>

                        <div className="grid gap-3 sm:grid-cols-2">
                          <div className="space-y-2">
                            <Label htmlFor="default-temperature">Temperature</Label>
                            <Input
                              id="default-temperature"
                              type="number"
                              step="0.05"
                              min="0"
                              max="2"
                              value={cfg.temperature}
                              onChange={(event) => setCfg({ ...cfg, temperature: parseFloat(event.target.value) })}
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="default-max-tokens">输出限制</Label>
                            <Input
                              id="default-max-tokens"
                              type="number"
                              step="100"
                              min="0"
                              value={cfg.max_tokens}
                              onChange={(event) => setCfg({ ...cfg, max_tokens: parseInt(event.target.value || "0", 10) })}
                            />
                          </div>
                        </div>

                        <Alert>
                          <AlertDescription className="space-y-2">
                            <Label htmlFor="default-json-format" className="flex items-start gap-2 font-normal">
                              <Checkbox
                                id="default-json-format"
                                checked={cfg.use_json_format}
                                onCheckedChange={(checked) => setCfg({ ...cfg, use_json_format: checked === true })}
                              />
                              <span>请求结构化 JSON 输出。实际格式由标准适配器处理。</span>
                            </Label>
                            <p className="text-xs leading-5 text-muted-foreground">输出限制为 0 时使用后端默认策略。</p>
                          </AlertDescription>
                        </Alert>
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>

                {err && (
                  <Alert variant="destructive">
                    <AlertDescription>{err}</AlertDescription>
                  </Alert>
                )}
              </div>
            </ScrollArea>
          </CardContent>
          <CardFooter className="shrink-0 bg-card px-4 py-3">
            <Button className="w-full gap-2" disabled={busy} onClick={submit}>
              {busy ? "创建中..." : "创建房间"}
              <ArrowRight className="size-4" />
            </Button>
          </CardFooter>
        </Card>
      </aside>
    </div>
  );
}

function InfoLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border bg-background/60 px-3 py-2 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="min-w-0 truncate text-right font-medium">{value}</span>
    </div>
  );
}
