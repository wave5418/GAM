import json
from typing import Dict, List, Optional, Union
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        """Get the appropriate answer based on category."""
        if self.category == 5:
            return self.adversarial_answer
        return self.answer

@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str

@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]

@dataclass
class Conversation:
    speaker_a: str
    speaker_b: str
    sessions: Dict[int, Session]

@dataclass
class EventSummary:
    events: Dict[str, Dict[str, List[str]]]  # session -> speaker -> events

@dataclass
class Observation:
    observations: Dict[str, Dict[str, List[List[str]]]]  # session -> speaker -> [observation, evidence]

@dataclass
class LoCoMoSample:
    """A single sample from the LoComo dataset"""
    sample_id: str
    qa: List[QA]
    conversation: Conversation
    event_summary: EventSummary
    observation: Observation
    session_summary: Dict[str, str]


def _safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _to_trajectory_steps(sample: dict) -> List[dict]:
    """
    Extract a trajectory list from common ALFWorld-style schemas.
    """
    if isinstance(sample.get("trajectory"), list):
        return sample["trajectory"]
    if isinstance(sample.get("steps"), list):
        return sample["steps"]
    if isinstance(sample.get("history"), list):
        return sample["history"]
    return []


def _build_turns_from_alfworld(sample: dict, sample_idx: int) -> List[Turn]:
    turns: List[Turn] = []
    trajectory = _to_trajectory_steps(sample)
    turn_idx = 1

    for step in trajectory:
        if not isinstance(step, dict):
            continue

        observation = (
            step.get("observation")
            or step.get("obs")
            or step.get("state")
            or step.get("description")
        )
        action = step.get("action") or step.get("command")

        if observation:
            turns.append(
                Turn(
                    speaker="Environment",
                    dia_id=f"{sample_idx}:{turn_idx}",
                    text=_safe_str(observation),
                )
            )
            turn_idx += 1

        if action:
            turns.append(
                Turn(
                    speaker="Agent",
                    dia_id=f"{sample_idx}:{turn_idx}",
                    text=_safe_str(action),
                )
            )
            turn_idx += 1

    # Fallback: some dumps only contain a plain text transcript
    if not turns:
        transcript = sample.get("transcript") or sample.get("text")
        if transcript:
            turns.append(
                Turn(
                    speaker="Environment",
                    dia_id=f"{sample_idx}:1",
                    text=_safe_str(transcript),
                )
            )

    return turns


def _build_qa_from_alfworld(sample: dict) -> List[QA]:
    qa_items: List[QA] = []

    # If dataset already has QA list, reuse it directly.
    raw_qas = sample.get("qa") or sample.get("qas") or []
    if isinstance(raw_qas, list):
        for qa in raw_qas:
            if not isinstance(qa, dict):
                continue
            question = qa.get("question") or qa.get("q")
            answer = qa.get("answer") or qa.get("a") or qa.get("gold_answer")
            if question and answer is not None:
                qa_items.append(
                    QA(
                        question=_safe_str(question),
                        answer=_safe_str(answer),
                        evidence=[],
                        category=int(qa.get("category", 4)) if str(qa.get("category", "4")).isdigit() else 4,
                    )
                )

    if qa_items:
        return qa_items

    # Otherwise generate a minimal factual QA from typical ALFWorld fields.
    task_goal = (
        sample.get("goal")
        or sample.get("task")
        or sample.get("task_description")
        or sample.get("instruction")
        or sample.get("mission")
    )

    if task_goal:
        qa_items.append(
            QA(
                question="What is the task goal in this ALFWorld episode?",
                answer=_safe_str(task_goal),
                evidence=[],
                category=4,
            )
        )

    final_answer = sample.get("final_answer") or sample.get("answer")
    if final_answer:
        qa_items.append(
            QA(
                question="What is the final outcome of this ALFWorld episode?",
                answer=_safe_str(final_answer),
                evidence=[],
                category=4,
            )
        )

    # Ensure at least one QA exists so evaluation can run.
    if not qa_items:
        qa_items.append(
            QA(
                question="What happened in this ALFWorld episode?",
                answer="No explicit answer provided in source data.",
                evidence=[],
                category=4,
            )
        )

    return qa_items

def parse_session(session_data: List[dict], session_id: int, date_time: str) -> Session:
    """Parse a single session's data, including turns with images by using their captions."""
    turns = []
    for turn in session_data:
        # For turns with images, combine caption and text
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text
            
        turns.append(Turn(
            speaker=turn["speaker"],
            dia_id=turn["dia_id"],
            text=text
        ))
    return Session(session_id=session_id, date_time=date_time, turns=turns)

def parse_conversation(conv_data: dict) -> Conversation:
    """Parse conversation data."""
    sessions = {}
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(value, session_id, date_time)
                if session.turns:
                    sessions[session_id] = session
    
    return Conversation(
        speaker_a=conv_data["speaker_a"],
        speaker_b=conv_data["speaker_b"],
        sessions=sessions
    )

def load_locomo_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
    """
    Load the LoComo dataset from a JSON file, including image-based content by using captions.
    
    Args:
        file_path: Path to the JSON file containing the dataset
        
    Returns:
        List of LoCoMoSample objects containing the parsed data
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
        
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    total_qa = 0
    total_image_qa = 0
    qa_counts_per_sample = []
    
    for sample_idx, sample in enumerate(data):
        try:
            qa_list = []
            sample_qa_count = 0
            sample_image_qa_count = 0
            
            for qa_idx, qa in enumerate(sample["qa"]):
                try:
                    has_image_evidence = False
                    for evidence_id in qa.get("evidence", []):
                        if ":" not in evidence_id:
                            continue
                        turn_id = evidence_id.split(":")[1]
                        for session in sample["conversation"].values():
                            if isinstance(session, list):
                                for turn in session:
                                    if turn.get("dia_id", "").endswith(turn_id):
                                        if "img_url" in turn or "blip_caption" in turn:
                                            has_image_evidence = True
                                            break
                    
                    if has_image_evidence:
                        sample_image_qa_count += 1
                        
                    qa_obj = QA(
                        question=qa["question"],
                        answer=qa.get("answer"),
                        evidence=qa.get("evidence", []),
                        category=qa.get("category"),
                        adversarial_answer=qa.get("adversarial_answer")
                    )
                    qa_list.append(qa_obj)
                    sample_qa_count += 1
                    
                except KeyError as e:
                    print(f"Error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e
                except Exception as e:
                    print(f"Unexpected error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e
            
            conversation = parse_conversation(sample["conversation"])
            
            event_summary = EventSummary(events=sample["event_summary"])
            
            observation = Observation(observations=sample["observation"])
            
            session_summary = sample.get("session_summary", {})
            
            sample_obj = LoCoMoSample(
                sample_id=str(sample_idx),
                qa=qa_list,
                conversation=conversation,
                event_summary=event_summary,
                observation=observation,
                session_summary=session_summary
            )
            samples.append(sample_obj)
            
            total_qa += sample_qa_count
            total_image_qa += sample_image_qa_count
            qa_counts_per_sample.append(sample_qa_count)
            
            print(f"\nSample {sample_idx}:")
            print(f"  Total QAs: {sample_qa_count}")
            print(f"  QAs with image evidence: {sample_image_qa_count}")
            
        except Exception as e:
            print(f"Error processing sample {sample_idx}:")
            print(str(e))
            raise e
    
    print("\nOverall Statistics:")
    print(f"Total QAs: {total_qa}")
    print(f"Total QAs with image evidence: {total_image_qa}")
    print(f"Average QAs per sample: {total_qa / len(samples):.2f}")
    print(f"Min QAs in a sample: {min(qa_counts_per_sample)}")
    print(f"Max QAs in a sample: {max(qa_counts_per_sample)}")
    
    return samples


def load_alfworld_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
    """
    Load ALFWorld-style trajectories and adapt them into the internal sample schema.

    Supported top-level layouts:
    - JSON list of episodes
    - JSON object with `episodes` list
    - JSON Lines (one episode per line)
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    content = file_path.read_text(encoding='utf-8').strip()
    episodes: List[dict] = []

    if not content:
        return []

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            episodes = parsed
        elif isinstance(parsed, dict):
            episodes = parsed.get("episodes", [])
        else:
            episodes = []
    except json.JSONDecodeError:
        # JSONL fallback
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            episodes.append(json.loads(line))

    samples: List[LoCoMoSample] = []
    for idx, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            continue

        turns = _build_turns_from_alfworld(episode, idx)
        qa_list = _build_qa_from_alfworld(episode)

        session = Session(
            session_id=1,
            date_time=_safe_str(
                episode.get("date_time")
                or episode.get("timestamp")
                or "2024-01-01 00:00:00"
            ),
            turns=turns,
        )

        conversation = Conversation(
            speaker_a="Agent",
            speaker_b="Environment",
            sessions={1: session},
        )

        session_summary_text = (
            _safe_str(episode.get("summary"))
            or _safe_str(episode.get("goal"))
            or _safe_str(episode.get("task"))
            or "ALFWorld episode summary unavailable."
        )

        samples.append(
            LoCoMoSample(
                sample_id=_safe_str(episode.get("sample_id") or episode.get("episode_id") or idx),
                qa=qa_list,
                conversation=conversation,
                event_summary=EventSummary(events={}),
                observation=Observation(observations={}),
                session_summary={"session_1_summary": session_summary_text},
            )
        )

    print(f"Loaded ALFWorld-style episodes: {len(samples)}")
    return samples

def get_dataset_statistics(samples: List[LoCoMoSample]) -> Dict:
    """
    Get basic statistics about the text-only dataset.
    
    Args:
        samples: List of LoCoMoSample objects
        
    Returns:
        Dictionary containing various statistics about the dataset
    """
    stats = {
        "num_samples": len(samples),
        "total_qa_pairs": sum(len(sample.qa) for sample in samples),
        "total_sessions": sum(len(sample.conversation.sessions) for sample in samples),
        "total_turns": sum(
            sum(len(session.turns) for session in sample.conversation.sessions.values())
            for sample in samples
        ),
        "qa_with_adversarial": sum(
            sum(1 for qa in sample.qa if qa.adversarial_answer is not None)
            for sample in samples
        )
    }
    return stats

if __name__ == "__main__":
    dataset_path = Path(__file__).parent / "data" / "locomo10.json"
    try:
        print(f"Loading dataset from: {dataset_path}")
        samples = load_locomo_dataset(dataset_path)
        for sample_idx, sample in enumerate(samples):
            print(f"\nSample {sample_idx}:")
            for _,turns in sample.conversation.sessions.items():
                for turn in turns.turns:
                    print(turn)
                    break   
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
