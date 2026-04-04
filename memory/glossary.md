# PermitIQ Glossary

## Acronyms
| Term | Meaning |
|------|---------|
| ZBA | Zoning Board of Appeals — the Boston board that grants/denies variances |
| BOA | Board of Appeal — prefix for ZBA case numbers (e.g., BOA1776619) |
| BPDA | Boston Planning & Development Agency — provides zoning subdistrict data |
| FAR | Floor Area Ratio — building size relative to lot size |
| AUC | Area Under the ROC Curve — ML model performance metric |
| SHAP | SHapley Additive exPlanations — ML model explainability |
| GCOD | Groundwater Conservation Overlay District |
| CFROD | Coastal Flood Resilience Overlay District |
| OCR | Optical Character Recognition — used to extract text from ZBA decision PDFs |
| TE | Target Encoding — encoding categorical variables using outcome rates |
| CV | Cross-Validation — model evaluation technique |
| PA | Property Assessment — Boston tax assessment data (FY2026, 184K records) |

## Key Concepts
| Term | Meaning |
|------|---------|
| Variance | Permission to deviate from zoning rules (e.g., build taller than allowed) |
| Conditional Use | A use not allowed as-of-right but grantable by ZBA |
| Proviso | Conditions attached to an approval (e.g., "must add landscaping") |
| Subdistrict | Granular zoning area within a broader district (286 unique in Boston) |
| Parcel-level variance | A variance required by the parcel's physical characteristics, regardless of proposal |
| Proposal-level variance | A variance triggered by the specific development proposal |
| Honest CV | Cross-validation that recomputes target encoding within each fold (prevents leakage) |

## Demo Addresses
| Address | Notes |
|---------|-------|
| 57 Centre St | Michael Winston's case, 1 case, small lot (1,200 sf) |
| 75 Tremont St | 14 cases, 62% approval |
| 1081 River St | 13 cases, 73% approval |
| 58 Burbank St | 10 cases, 44% approval |
