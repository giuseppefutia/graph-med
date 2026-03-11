
from datetime import date
from typing import Any, Dict, Literal, List, Optional
from pydantic import BaseModel, Field

###############################
### Ontology Mapping Models ###
###############################

class OntologyMappingInput(BaseModel):
    source_concept: str
    source_context: str
    candidate_list: str

class SupportItem(BaseModel):
    evidence: str
    reason: str

class OntologyMappingResponse(BaseModel):
    best_id: str
    best_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    support: SupportItem

##########################
### NER Patient Models ###
##########################

class PatientNERInput(BaseModel):
    icd_chapters: List[str]
    patient_id: str
    encounter_id: str
    concat_text: str
    narrative_text: str

class PatientNEREntity(BaseModel):
    source: str        # "concat" or "narrative"
    start: int         # 0-based [start, end)
    end: int
    text: str
    label: str         # must be one of icd_chapters
    assertion: str     # "present" | "negated" | "uncertain"
    temporality: str   # "acute" | "chronic" | "recurrent" | "history" | "unspecified"
    rationale: str

class PatientNERResponse(BaseModel):
    patient_id: str
    encounter_id: str
    entities: List[PatientNEREntity] = Field(default_factory=list)

###################
### Patient NED ###
###################

class PatientNEDCandidate(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    id: str                 # e.g., "F32.0"
    label: str              # e.g., "Mild depressive episode"

class PatientNEDOtherMention(BaseModel):
    text: str
    label: str              # ICD Chapter label for the other mention

class PatientNEDInput(BaseModel):
    mention: PatientNEREntity
    candidates: List[PatientNEDCandidate]
    other_mentions: List[PatientNEDOtherMention] = Field(default_factory=list)

class PatientNEDResponse(PatientNEREntity):
    icd_id: Optional[str] = None        # e.g., "R45.2"; None if abstaining
    icd_label: Optional[str] = None     # e.g., "Unhappiness"; None if abstaining
    confidence: float = Field(default=None, ge=0.0, le=1.0)
    linking_rationale: str

####################
### Query Models ###
####################

class GuardrailsDecision(BaseModel):
    decision: Literal["continue", "end"]
    reason: Optional[str] = None
    
class ValidationError(BaseModel):
    type: Literal["syntax", "label", "relationship", "property", "variable", "semantic", "structure"]
    message: str
    suggestion: Optional[str] = ""

class ValidateCypherOutput(BaseModel):
    errors: List[ValidationError] = Field(default_factory=list)

class DiagnoseCypherOutput(BaseModel):
    issues: List[str]
    suggestions: List[str]
    fixed_cypher: Optional[str] = None

class GeneralMedicalInput(BaseModel):
    question: str
    top_k: Optional[int] = 20

class GeneralMedicalResponse(BaseModel):
    cypher: str
    rows: List[Dict[str, Any]]
    steps: List[str] = []
    explanation: str

######################
### Patient Models ###
######################

class PatientInfoInput(BaseModel):
    patient_id: str
    question: str
    # Optional selector: ISO date (YYYY-MM-DD) or "latest"
    encounter_date: Optional[str] = Field(
        default=None,
        description=(
            "Optional encounter selector. "
            "Use 'YYYY-MM-DD' for a specific Encounter.period.start date, "
            "or 'latest' to use the most recent encounter. "
            "If omitted, all encounters will be included."
        ),
    )

class PatientCoverageInput(BaseModel):
    patient_id: str
    limit: int = 20

class CoverageRow(BaseModel):
    diseaseId: str
    diseaseName: str
    covered: int
    total: int
    coveragePct: float
    missingHpoIds: List[str]

class CoverageResponse(BaseModel):
    cypher: str
    rows: List[CoverageRow]
    steps: List[str] = []

class PatientCoverageResponse(BaseModel):
    cypher: str
    rows: List[CoverageRow]
    steps: List[str] = []
