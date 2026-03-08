# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Egosocial environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import EgosocialAction, EgosocialObservation
except ImportError:  # pragma: no cover
    from models import EgosocialAction, EgosocialObservation
from .egosocial_env_environment import EgosocialEnvironment


app = create_app(
    EgosocialEnvironment,
    EgosocialAction,
    EgosocialObservation,
    env_name="egosocial_env",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Run the local development server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
