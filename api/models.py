from pydantic import BaseModel
from typing import Optional

#defines what a developer submits
class ProposalRequest(BaseModel):
    parcel_id: str
    proposed_use: Optional[str] = None
    proposed_units: Optional[int] = None
    proposed_height: Optional[float] = None
    proposed_far: Optional[float] = None
    proposed_lot_coverage: Optional[float] = None
