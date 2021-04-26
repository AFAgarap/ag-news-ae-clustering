from pydantic import BaseModel


class ClusterRequest(BaseModel):
    text: str


class ClusterResponse(BaseModel):
    text: str
    cluster_index: int
