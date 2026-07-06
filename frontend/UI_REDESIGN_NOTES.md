# Werewolf MAS UI redesign notes

This UI is designed around a spectator/god-view workflow, not a generic
dashboard. The core screen should keep three stable areas visible on desktop:

- Player seats: who is alive, who is speaking, role visibility, and vote/death
  state.
- Public testimony: the primary reading surface for speeches, last words,
  votes, and AI reasoning paired with the final content.
- Live context: phase, vote progress, recent events, win conditions, and end
  summary.

Research references used for the redesign direction:

- Material Design navigation rail and adaptive layout guidance:
  https://m3.material.io/components/navigation-rail/overview and
  https://m3.material.io/foundations/layout/applying-layout/window-size-classes.
  Stable side
  regions are appropriate on larger screens when users need persistent context.
- Apple Human Interface Guidelines sidebars:
  https://developer.apple.com/design/human-interface-guidelines/sidebars.
  Sidebars work well for persistent
  navigation/context without stealing the primary content surface.
- IBM progressive disclosure guidance:
  https://www.ibm.com/docs/en/technical-content?topic=practices-progressive-disclosure.
  Advanced/contextual details should be
  grouped and revealed in-place instead of competing with primary content.
- Conversation UI guidance:
  https://www.aiuxdesign.guide/guides/conversational-ui-guide/anatomy-of-a-chat-interface.
  Messages should keep speaker, content, and
  supporting context together, while avoiding clutter in the primary reading
  path.
- Atlassian layout system:
  https://atlassian.design/components/navigation-system/layout.
  App layouts should distinguish navigation/context regions from the primary
  content area.

Implementation principles:

- The center column is the primary surface. It must not be displaced by replay,
  metrics, or player lists.
- New messages scroll the chat viewport only; page scroll position must remain
  stable.
- AI thinking and final content belong to the same message card.
- System events should read as timeline separators, not full message cards.
- Votes, accusations, support, opposition, and observations are secondary
  message signals and should not interrupt the speech body.
- End-of-game analysis belongs in the live context panel or replay mode; it
  must not collapse the active spectator layout.
