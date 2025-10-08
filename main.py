"""Test .py file to learn pydantic-ai"""

import asyncio
from dataclasses import dataclass
from typing import Annotated, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

# Load environment variables from a .env file
load_dotenv()


# Mock database
@dataclass
class Patient:
    """A patient in the database."""

    id: int
    name: str
    age: int
    vitals: dict[str, Any]


PATIENT_DB = {
    1: Patient(
        id=1, name="John Doe", age=30, vitals={"heart_rate": 72, "bp": "120/80"}
    ),
    2: Patient(
        id=2, name="Jane Smith", age=25, vitals={"heart_rate": 75, "bp": "110/70"}
    ),
}


# Helper class for the database operations
class Database:
    """Simulated async database operations."""

    @staticmethod
    async def patient_name(patient_id: int) -> str:
        """Get patient name by ID."""
        await asyncio.sleep(0.1)  # Simulate async operation
        patient = PATIENT_DB.get(patient_id)
        if not patient:
            raise ValueError("Patient not found")
        return patient.name

    @staticmethod
    async def patient_age(patient_id: int) -> int:
        """Get patient age by ID."""
        await asyncio.sleep(0.1)  # Simulate async operation
        patient = PATIENT_DB.get(patient_id)
        if not patient:
            raise ValueError("Patient not found")
        return patient.age

    @staticmethod
    async def patient_vitals(patient_id: int) -> dict[str, Any]:
        """Get patient vitals by ID."""
        await asyncio.sleep(0.1)  # Simulate async operation
        patient = PATIENT_DB.get(patient_id)
        if not patient:
            raise ValueError("Patient not found")
        return patient.vitals


@dataclass
class TriageDependencies:
    """Dependencies for triage operations."""

    patient_id: int
    db: Database


class TriageOutput(BaseModel):
    """Output model for triage operations."""

    response_text: Annotated[str, Field(description="Message to the patient")]
    escalate: Annotated[
        bool, Field(description="Whether to escalate the case to a human nurse")
    ]
    urgency: Annotated[int, Field(description="Urgency level of the case")]


triage_agent = Agent(
    model="gpt-5-nano",
    deps_type=TriageDependencies,
    output_type=TriageOutput,
    system_prompt=(
        "You are a triage assistant helping patients."
        "Provide clear advice and assess urgency."
    ),
)


@triage_agent.tool
async def get_patient_name(ctx: RunContext[TriageDependencies]) -> str:
    """Return the patient's name for this run."""
    return await ctx.deps.db.patient_name(ctx.deps.patient_id)


@triage_agent.tool
async def get_patient_age(ctx: RunContext[TriageDependencies]) -> int:
    """Return the patient's age for this run."""
    return await ctx.deps.db.patient_age(ctx.deps.patient_id)


@triage_agent.tool
async def get_patient_vitals(ctx: RunContext[TriageDependencies]) -> dict[str, Any]:
    """Return the patient's vitals for this run."""
    return await ctx.deps.db.patient_vitals(ctx.deps.patient_id)


async def main() -> None:
    """Main function to test database operations."""
    db = Database()
    patient_id = 1

    name = await db.patient_name(patient_id)
    age = await db.patient_age(patient_id)
    vitals = await db.patient_vitals(patient_id)

    print(f"Patient ID: {patient_id}")
    print(f"Name: {name}")
    print(f"Age: {age}")
    print(f"Vitals: {vitals}")

    deps = TriageDependencies(patient_id=patient_id, db=db)

    result = await triage_agent.run(
        "The patient is experiencing mild chest pain and shortness of breath.",
        deps=deps,
    )

    # Print the structured output
    print("\n______________________________________________________")
    print("\nTriage Agent Response:")
    print("\n______________________________________________________")
    print(f"Response Text: {result.output.response_text}")
    print("\n______________________________________________________")
    print(f"Escalate: {result.output.escalate}")
    print("\n______________________________________________________")
    print(f"Urgency: {result.output.urgency}")


if __name__ == "__main__":
    asyncio.run(main())
