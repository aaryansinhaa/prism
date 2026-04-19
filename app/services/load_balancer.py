"""Load balancing service for distributing requests across multiple container instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import httpx
from starlette import status


@dataclass
class ContainerInstance:
    """Represents a single container instance for a model."""

    container_id: str
    port: int
    instance_index: int = 0
    healthy: bool = True
    consecutive_failures: int = 0
    max_consecutive_failures: int = 3

    def get_url(self) -> str:
        """Get the base URL for this container."""
        return f"http://127.0.0.1:{self.port}"

    def mark_failure(self) -> None:
        """Mark a failure on this instance."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.healthy = False

    def mark_success(self) -> None:
        """Mark a successful request on this instance."""
        self.consecutive_failures = 0
        self.healthy = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "container_id": self.container_id,
            "port": self.port,
            "instance_index": self.instance_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContainerInstance:
        """Create from dictionary."""
        return cls(
            container_id=data["container_id"],
            port=data["port"],
            instance_index=data.get("instance_index", 0),
        )


@dataclass
class LoadBalancerState:
    """Tracks load balancing state for a model version."""

    model_id: str
    version: str
    instances: List[ContainerInstance] = field(default_factory=list)
    current_index: int = 0

    def add_instance(
        self,
        container_id: str,
        port: int,
        instance_index: int = 0,
    ) -> ContainerInstance:
        """Add a new container instance."""
        instance = ContainerInstance(
            container_id=container_id,
            port=port,
            instance_index=instance_index,
        )
        self.instances.append(instance)
        return instance

    def get_next_instance(self) -> ContainerInstance | None:
        """Get next healthy instance using round-robin."""
        if not self.instances:
            return None

        # Filter healthy instances
        healthy = [i for i in self.instances if i.healthy]
        if not healthy:
            # If no healthy instances, try all (they may recover)
            healthy = self.instances

        # Round-robin: cycle through healthy instances
        self.current_index = self.current_index % len(healthy)
        instance = healthy[self.current_index]
        self.current_index = (self.current_index + 1) % len(healthy)
        return instance

    def mark_instance_success(self, instance: ContainerInstance) -> None:
        """Mark an instance as having successful request."""
        instance.mark_success()

    def mark_instance_failure(self, instance: ContainerInstance) -> None:
        """Mark an instance as having failed request."""
        instance.mark_failure()

    def get_healthy_count(self) -> int:
        """Get count of healthy instances."""
        return sum(1 for i in self.instances if i.healthy)

    def get_total_count(self) -> int:
        """Get total instance count."""
        return len(self.instances)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "instances": [i.to_dict() for i in self.instances],
            "current_index": self.current_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LoadBalancerState:
        """Create from dictionary."""
        state = cls(
            model_id=data["model_id"],
            version=data["version"],
            current_index=data.get("current_index", 0),
        )
        for inst_data in data.get("instances", []):
            state.instances.append(ContainerInstance.from_dict(inst_data))
        return state


class LoadBalancer:
    """Manages load balancing across multiple container instances."""

    def __init__(self):
        """Initialize load balancer."""
        self._state: Dict[str, Dict[str, LoadBalancerState]] = {}

    def register_instance(
        self,
        model_id: str,
        version: str,
        container_id: str,
        port: int,
        instance_index: int = 0,
    ) -> LoadBalancerState:
        """Register a new container instance for a model version."""
        if model_id not in self._state:
            self._state[model_id] = {}

        key = version or "v1"
        if key not in self._state[model_id]:
            self._state[model_id][key] = LoadBalancerState(
                model_id=model_id, version=key
            )

        lb_state = self._state[model_id][key]
        lb_state.add_instance(container_id, port, instance_index)
        return lb_state

    def get_state(
        self, model_id: str, version: str | None = None
    ) -> LoadBalancerState | None:
        """Get load balancer state for a model version."""
        if model_id not in self._state:
            return None
        key = version or "v1"
        return self._state[model_id].get(key)

    def get_next_instance(
        self, model_id: str, version: str | None = None
    ) -> ContainerInstance | None:
        """Get next instance for request using round-robin."""
        state = self.get_state(model_id, version)
        if not state:
            return None
        return state.get_next_instance()

    def mark_success(
        self, model_id: str, instance: ContainerInstance, version: str | None = None
    ) -> None:
        """Mark successful request on instance."""
        state = self.get_state(model_id, version)
        if state:
            state.mark_instance_success(instance)

    def mark_failure(
        self, model_id: str, instance: ContainerInstance, version: str | None = None
    ) -> None:
        """Mark failed request on instance."""
        state = self.get_state(model_id, version)
        if state:
            state.mark_instance_failure(instance)

    def get_health_summary(
        self, model_id: str, version: str | None = None
    ) -> Dict[str, Any] | None:
        """Get health summary for a model version."""
        state = self.get_state(model_id, version)
        if not state:
            return None
        return {
            "model_id": model_id,
            "version": state.version,
            "total_instances": state.get_total_count(),
            "healthy_instances": state.get_healthy_count(),
            "instances": [
                {
                    "container_id": i.container_id,
                    "port": i.port,
                    "healthy": i.healthy,
                    "failures": i.consecutive_failures,
                }
                for i in state.instances
            ],
        }


# Global load balancer instance
_load_balancer = LoadBalancer()


def get_load_balancer() -> LoadBalancer:
    """Get the global load balancer instance."""
    return _load_balancer


def reset_load_balancer() -> None:
    """Reset load balancer state (for testing)."""
    global _load_balancer
    _load_balancer = LoadBalancer()
