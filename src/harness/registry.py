"""Explicit, exact-version registry for environment plugins."""
from __future__ import annotations

from importlib import metadata
from typing import Iterable

from .environment import (
    ENVIRONMENT_PLUGIN_API_VERSION,
    EnvironmentDescriptor,
    EnvironmentPlugin,
)


class EnvironmentRegistryError(LookupError):
    pass


class EnvironmentRegistry:
    def __init__(self) -> None:
        self._plugins: dict[tuple[str, str], EnvironmentPlugin] = {}

    def register(self, plugin: EnvironmentPlugin) -> EnvironmentDescriptor:
        try:
            descriptor = EnvironmentDescriptor.model_validate(plugin.descriptor)
        except (AttributeError, TypeError, ValueError) as err:
            raise EnvironmentRegistryError("environment plugin descriptor is invalid") from err
        if descriptor.plugin_api_version != ENVIRONMENT_PLUGIN_API_VERSION:
            raise EnvironmentRegistryError(
                "environment plugin API mismatch: "
                f"{descriptor.plugin_api_version!r} != {ENVIRONMENT_PLUGIN_API_VERSION!r}"
            )
        key = (descriptor.id, descriptor.version)
        if key in self._plugins:
            raise EnvironmentRegistryError(
                f"environment plugin already registered: {descriptor.id}@{descriptor.version}"
            )
        contract = getattr(plugin, "decision_contract", None)
        if (
            contract is None
            or not isinstance(getattr(contract, "envelope_type", None), type)
            or not callable(getattr(contract, "validate_envelope", None))
        ):
            raise EnvironmentRegistryError("environment plugin decision contract is invalid")
        if not callable(getattr(plugin, "resolve_config", None)):
            raise EnvironmentRegistryError("environment plugin has no resolve_config method")
        if not callable(getattr(plugin, "create_session", None)):
            raise EnvironmentRegistryError("environment plugin has no create_session method")
        self._plugins[key] = plugin
        return descriptor

    def get(self, environment_id: str, version: str) -> EnvironmentPlugin:
        key = (str(environment_id), str(version))
        try:
            return self._plugins[key]
        except KeyError as err:
            raise EnvironmentRegistryError(
                f"unknown environment plugin: {key[0]}@{key[1]}"
            ) from err

    def descriptors(self) -> list[EnvironmentDescriptor]:
        return [
            EnvironmentDescriptor.model_validate(plugin.descriptor)
            for _key, plugin in sorted(self._plugins.items())
        ]

    def load_entry_points(
        self,
        *,
        names: Iterable[str] | None = None,
        group: str = "agent_harness.environments",
    ) -> list[EnvironmentDescriptor]:
        """Explicitly load selected third-party plugins.

        Loading an entry point executes third-party Python code. Callers must
        opt in by invoking this method; the registry never scans automatically.
        """
        selected_names = set(names) if names is not None else None
        loaded: list[EnvironmentDescriptor] = []
        for entry_point in metadata.entry_points().select(group=group):
            if selected_names is not None and entry_point.name not in selected_names:
                continue
            candidate = entry_point.load()
            plugin = candidate() if isinstance(candidate, type) else candidate
            loaded.append(self.register(plugin))
        return loaded
