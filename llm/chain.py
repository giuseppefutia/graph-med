from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm.prompt import (
    ONTOLOGY_MAPPING_PROMPT,
    PATIENT_NER_PROMPT,
    PATIENT_NED_PROMPT,
    GUARDRAILS_PROMPT,
    TEXT_2_CYPHER_PROMPT,
    QUERY_VALIDATION_PROMPT,
    DIAGNOSE_CYPHER_PROMPT,
    QUERY_CORRECTION_PROMPT,
    CLINICIAN_EXPLANATION_PROMPT,
    PATIENT_EXPLANATION_PROMPT,
    PATIENT_COVERAGE_PROMPT,
    FINAL_ANSWER_PROMPT,
)

from llm.pydantic_model import (
    OntologyMappingResponse,
    PatientNERResponse,
    PatientNEDResponse,
    GuardrailsDecision,
    ValidateCypherOutput,
    DiagnoseCypherOutput,
    PatientCoverageResponse
)

#######################
### Building chains ###
#######################


def ontology_mapping_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            ONTOLOGY_MAPPING_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            ONTOLOGY_MAPPING_PROMPT["user"],
            template_format="jinja2",
        ),
    ])
    return prompt | llm_model.with_structured_output(OntologyMappingResponse)


def patient_ner_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            PATIENT_NER_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            PATIENT_NER_PROMPT["user"],
            template_format="jinja2",
        ),
    ])
    return prompt | llm_model.with_structured_output(PatientNERResponse)


def patient_ned_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            PATIENT_NED_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            PATIENT_NED_PROMPT["user"],
            template_format="jinja2",
        ),
    ])
    return prompt | llm_model.with_structured_output(PatientNEDResponse)


########################
### Retrieval chains ###
########################


def get_guardrails_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            GUARDRAILS_PROMPT["system"], template_format="jinja2"
        ),
        HumanMessagePromptTemplate.from_template(
            GUARDRAILS_PROMPT["user"], template_format="jinja2"
        ),
    ])
    return prompt | llm_model.with_structured_output(GuardrailsDecision)


def text2cypher_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            TEXT_2_CYPHER_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            TEXT_2_CYPHER_PROMPT["user"],
            template_format="jinja2",
        ),
    ])
    return prompt | llm_model | StrOutputParser()


def validate_cypher_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            QUERY_VALIDATION_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            QUERY_VALIDATION_PROMPT["user"],
            template_format="jinja2"
        )
    ])
    return prompt | llm_model.with_structured_output(ValidateCypherOutput)


def diagnose_cypher_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            DIAGNOSE_CYPHER_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            DIAGNOSE_CYPHER_PROMPT["user"],
            template_format="jinja2",
        ),
    ])
    return prompt | llm_model.with_structured_output(DiagnoseCypherOutput)


def correct_cypher_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            QUERY_CORRECTION_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            QUERY_CORRECTION_PROMPT["user"],
            template_format="jinja2"
        )
    ])
    return prompt | llm_model | StrOutputParser()


def clinician_explanation_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            CLINICIAN_EXPLANATION_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            CLINICIAN_EXPLANATION_PROMPT["user"],
            template_format="jinja2",
        ),
    ])
    return prompt | llm_model | StrOutputParser()


def get_patient_answer_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            PATIENT_EXPLANATION_PROMPT["system"],
            template_format="jinja2",
        ),
        HumanMessagePromptTemplate.from_template(
            PATIENT_EXPLANATION_PROMPT["user"],
            template_format="jinja2",
        ),
    ])
    return prompt | llm_model | StrOutputParser()


def patient_coverage_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            PATIENT_COVERAGE_PROMPT["system"],
            template_format="jinja2"
        ),
        HumanMessagePromptTemplate.from_template(
            PATIENT_COVERAGE_PROMPT["user"],
            template_format="jinja2"
            ),
        ])
    return prompt | llm_model.with_structured_output(PatientCoverageResponse)


def get_final_answer_chain(llm_model: ChatOpenAI):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            FINAL_ANSWER_PROMPT["system"], template_format="jinja2"
        ),
        HumanMessagePromptTemplate.from_template(
            FINAL_ANSWER_PROMPT["user"], template_format="jinja2"
        ),
    ])
    return prompt | llm_model | StrOutputParser()
