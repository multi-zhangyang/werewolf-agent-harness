import * as React from "react"
import { ScrollArea as ScrollAreaPrimitive } from "radix-ui"

import { cn } from "@/lib/utils"

const RADIX_SCROLLAREA_STYLE_PREFIX = "[data-radix-scroll-area-viewport]"

function ScrollArea({
  className,
  viewportClassName,
  viewportRef,
  onViewportScroll,
  children,
  ...props
}: React.ComponentProps<typeof ScrollAreaPrimitive.Root> & {
  viewportClassName?: string
  viewportRef?: React.Ref<HTMLDivElement>
  onViewportScroll?: React.UIEventHandler<HTMLDivElement>
}) {
  const rootRef = React.useRef<HTMLDivElement>(null)

  React.useLayoutEffect(() => {
    const root = rootRef.current
    if (!root) return
    for (const styleNode of root.querySelectorAll("style")) {
      if (styleNode.textContent?.startsWith(RADIX_SCROLLAREA_STYLE_PREFIX)) {
        styleNode.remove()
      }
    }
  })

  return (
    <ScrollAreaPrimitive.Root
      ref={rootRef}
      data-slot="scroll-area"
      className={cn("relative overflow-hidden", className)}
      {...props}
    >
      <ScrollAreaPrimitive.Viewport
        ref={viewportRef}
        onScroll={onViewportScroll}
        data-slot="scroll-area-viewport"
        className={cn(
          "size-full min-w-0 rounded-[inherit] transition-[color,box-shadow] outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-1 [&>div]:!block [&>div]:!min-w-0 [&>div]:!w-full",
          viewportClassName
        )}
      >
        {children}
      </ScrollAreaPrimitive.Viewport>
      <ScrollBar />
      <ScrollAreaPrimitive.Corner />
    </ScrollAreaPrimitive.Root>
  )
}

function ScrollBar({
  className,
  orientation = "vertical",
  ...props
}: React.ComponentProps<typeof ScrollAreaPrimitive.ScrollAreaScrollbar>) {
  return (
    <ScrollAreaPrimitive.ScrollAreaScrollbar
      data-slot="scroll-area-scrollbar"
      data-orientation={orientation}
      orientation={orientation}
      className={cn(
        "flex touch-none p-px transition-colors select-none data-horizontal:h-2.5 data-horizontal:flex-col data-horizontal:border-t data-horizontal:border-t-transparent data-vertical:h-full data-vertical:w-2.5 data-vertical:border-l data-vertical:border-l-transparent",
        className
      )}
      {...props}
    >
      <ScrollAreaPrimitive.ScrollAreaThumb
        data-slot="scroll-area-thumb"
        className="relative flex-1 rounded-full bg-border"
      />
    </ScrollAreaPrimitive.ScrollAreaScrollbar>
  )
}

export { ScrollArea, ScrollBar }
