import { useState } from "react";
import { SlidersHorizontal } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ProviderMeta } from "../lib/types";

export function ModelConfigModal({
  seat,
  providers,
  initial,
  onClose,
  onSave,
}: {
  seat: number;
  providers: Record<string, ProviderMeta>;
  initial?: any;
  onClose: () => void;
  onSave: (cfg: any) => void;
}) {
  const providerEntries: [string, ProviderMeta][] = Object.entries(providers);
  const providerOptions: [string, ProviderMeta][] = providerEntries.length
    ? providerEntries
    : [["openai", { label: "OpenAI-compatible", hint: "", default_api_base: "", default_model: "" }]];
  const [cfg, setCfg] = useState<any>({
    provider: initial?.provider || providerEntries[0]?.[0] || "openai",
    model: initial?.model || "",
    api_base: initial?.api_base || "",
    api_key: "",
    temperature: initial?.temperature ?? 0.85,
    max_tokens: initial?.max_tokens ?? 0,
    use_json_format: initial?.use_json_format ?? false,
  });

  return (
    <Dialog open onOpenChange={(open) => { if (!open) onClose(); }}>
      <DialogContent className="flex max-h-[calc(100svh-2rem)] flex-col overflow-hidden sm:max-w-lg">
        <DialogHeader className="shrink-0 pr-8">
          <DialogTitle className="flex items-center gap-2">
            <SlidersHorizontal className="size-4" />
            {seat}号模型设置
          </DialogTitle>
          <DialogDescription>只配置标准兼容字段；留空会继承房间默认配置。</DialogDescription>
        </DialogHeader>

        <ScrollArea className="-mx-4 min-h-0 flex-1 px-4">
          <div className="space-y-4 pb-4">
            <div className="space-y-2">
              <Label htmlFor={`seat-${seat}-provider`}>Provider</Label>
              <Select value={cfg.provider} onValueChange={(provider) => setCfg({ ...cfg, provider })}>
                <SelectTrigger id={`seat-${seat}-provider`} className="w-full">
                  <SelectValue placeholder="选择协议/Provider" />
                </SelectTrigger>
                <SelectContent>
                  {providerOptions.map(([key, value]) => (
                    <SelectItem key={key} value={key}>{value.label || key}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor={`seat-${seat}-model`}>Model</Label>
                <Input
                  id={`seat-${seat}-model`}
                  value={cfg.model}
                  placeholder="留空继承"
                  onChange={(event) => setCfg({ ...cfg, model: event.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor={`seat-${seat}-temperature`}>Temperature</Label>
                <Input
                  id={`seat-${seat}-temperature`}
                  type="number"
                  step="0.05"
                  min="0"
                  max="2"
                  value={cfg.temperature}
                  onChange={(event) => setCfg({ ...cfg, temperature: parseFloat(event.target.value) })}
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor={`seat-${seat}-api-base`}>API Base</Label>
              <Input
                id={`seat-${seat}-api-base`}
                value={cfg.api_base}
                placeholder="留空继承"
                onChange={(event) => setCfg({ ...cfg, api_base: event.target.value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor={`seat-${seat}-api-key`}>API Key</Label>
              <Input
                id={`seat-${seat}-api-key`}
                type="password"
                value={cfg.api_key}
                placeholder="留空继承"
                onChange={(event) => setCfg({ ...cfg, api_key: event.target.value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor={`seat-${seat}-max-tokens`}>输出限制</Label>
              <Input
                id={`seat-${seat}-max-tokens`}
                type="number"
                step="100"
                min="0"
                value={cfg.max_tokens}
                onChange={(event) => setCfg({ ...cfg, max_tokens: parseInt(event.target.value || "0", 10) })}
              />
            </div>

            <Alert>
              <AlertDescription className="space-y-2">
                <Label htmlFor={`seat-${seat}-json-format`} className="flex items-start gap-2 font-normal">
                  <Checkbox
                    id={`seat-${seat}-json-format`}
                    checked={cfg.use_json_format}
                    onCheckedChange={(checked) => setCfg({ ...cfg, use_json_format: checked === true })}
                  />
                  <span>请求结构化 JSON 输出。实际请求格式由后端标准适配器决定。</span>
                </Label>
                <p className="text-xs leading-5 text-muted-foreground">
                  输出限制填 0 表示使用后端默认策略；UI 不为某个模型服务商做特殊分支。
                </p>
              </AlertDescription>
            </Alert>
          </div>
        </ScrollArea>

        <DialogFooter className="shrink-0">
          <Button variant="ghost" onClick={onClose}>取消</Button>
          <Button onClick={() => onSave(cfg)}>保存</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
