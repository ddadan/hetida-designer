import datetime
from typing import Any

from pydantic import BaseModel, Field  # pylint: disable=no-name-in-module

from demo_adapter_python.external_types import ExternalType, ValueDataType


class InfoResponse(BaseModel):
    id: str  # noqa: A003
    name: str
    version: str


class StructureThingNode(BaseModel):
    id: str  # noqa: A003
    parentId: str | None = None
    name: str
    description: str


class StructureFilter(BaseModel):
    name: str
    required: bool


class StructureSource(BaseModel):
    id: str  # noqa: A003
    thingNodeId: str
    name: str
    type: ExternalType  # noqa: A003
    visible: bool | None = True
    path: str
    metadataKey: str | None = None
    filters: dict[str, StructureFilter] | None = {}


class StructureSink(BaseModel):
    id: str  # noqa: A003
    thingNodeId: str
    name: str
    type: ExternalType  # noqa: A003
    visible: bool | None = True
    path: str
    metadataKey: str | None = None
    filters: dict[str, StructureFilter] | None = {}


class StructureResponse(BaseModel):
    id: str  # noqa: A003
    name: str
    thingNodes: list[StructureThingNode]
    sources: list[StructureSource]
    sinks: list[StructureSink]


class MultipleSourcesResponse(BaseModel):
    resultCount: int
    sources: list[StructureSource]


class MultipleSinksResponse(BaseModel):
    resultCount: int
    sinks: list[StructureSink]


class PostMetadatum(BaseModel):
    key: str
    value: Any = Field(..., example=True)


class Metadatum(BaseModel):
    key: str
    value: Any | None = None
    dataType: ValueDataType
    isSink: bool | None = False


class TimeseriesRecord(BaseModel):
    timestamp: datetime.datetime = Field(
        ..., example=datetime.datetime.now(datetime.timezone.utc)
    )
    value: Any = Field(..., example=0.25)
