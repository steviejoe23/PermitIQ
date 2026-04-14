"""
Board Member Routes — ZBA board member voting profiles and tendencies.

Extracted from hearing transcripts to show which members tend to approve
or deny specific variance types.
"""

import json
import logging
import os
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

logger = logging.getLogger("permitiq")
router = APIRouter(prefix="/board_members", tags=["Board Members"])

_profiles = None
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'board_member_profiles.json')


def init():
    global _profiles
    path = DATA_PATH
    if not os.path.exists(path):
        # Try relative to working directory
        path = os.path.join('data', 'board_member_profiles.json')
    if os.path.exists(path):
        with open(path) as f:
            _profiles = json.load(f)
        logger.info("Board member profiles loaded (%d members)", len(_profiles.get('members', [])))
    else:
        logger.warning("Board member profiles not found at %s", path)
        _profiles = {"members": []}


@router.get("")
def list_board_members():
    """List all known ZBA board members with summary stats."""
    if _profiles is None:
        init()
    members = _profiles.get('members', [])
    summary = []
    for m in members:
        summary.append({
            "name": m['name'],
            "role": m.get('role', 'Member'),
            "hearings_attended": m.get('hearings_attended', 0),
            "date_range": m.get('date_range', []),
            "approval_rate": m.get('approval_rate'),
            "denial_rate": m.get('denial_rate'),
        })
    summary.sort(key=lambda x: x['hearings_attended'], reverse=True)
    return {"members": summary, "total": len(summary)}


@router.get("/{member_name}/profile")
def board_member_profile(member_name: str):
    """Full voting profile for a board member including variance-specific tendencies."""
    if _profiles is None:
        init()
    members = _profiles.get('members', [])

    # Fuzzy match
    name_lower = member_name.lower().strip()
    match = None
    for m in members:
        if m['name'].lower() == name_lower:
            match = m
            break
        for alias in m.get('aliases', []):
            if alias.lower() == name_lower:
                match = m
                break
        if match:
            break

    # Try partial match
    if not match:
        for m in members:
            if name_lower in m['name'].lower() or m['name'].lower() in name_lower:
                match = m
                break

    if not match:
        raise HTTPException(status_code=404, detail=f"Board member '{member_name}' not found. Use /board_members to see all members.")

    return match


@router.get("/for_hearing")
def members_for_hearing(
    variance_types: Optional[str] = Query(None, description="Comma-separated variance types"),
):
    """Show board member tendencies relevant to specific variance types.

    Useful for hearing prep — shows which members are likely to be sympathetic
    or skeptical of specific variance requests.
    """
    if _profiles is None:
        init()
    members = _profiles.get('members', [])

    vtypes = []
    if variance_types:
        vtypes = [v.strip().lower() for v in variance_types.split(',') if v.strip()]

    results = []
    for m in members:
        var_stats = m.get('variance_stats', {})
        member_info = {
            "name": m['name'],
            "role": m.get('role', 'Member'),
            "hearings_attended": m.get('hearings_attended', 0),
            "overall_approval_rate": m.get('approval_rate'),
        }

        if vtypes:
            relevant_stats = {}
            for vt in vtypes:
                for key, stats in var_stats.items():
                    if vt in key.lower():
                        relevant_stats[key] = stats
            member_info["variance_tendencies"] = relevant_stats
        else:
            member_info["variance_tendencies"] = var_stats

        results.append(member_info)

    results.sort(key=lambda x: x['hearings_attended'], reverse=True)
    return {"members": results, "variance_filter": vtypes}
