import { useState } from "react";
import type { ClientMessage } from "../../ws/messageTypes";

interface ChoiceOption {
  label: string;
  effect_ir?: unknown;
}

interface Props {
  prompt: string;
  options: ChoiceOption[];
  minChoices: number;
  maxChoices: number;
  send: (msg: ClientMessage) => void;
  onDone: () => void;
}

export function ChoiceModal({ prompt, options, minChoices, maxChoices, send, onDone }: Props) {
  const [submitted, setSubmitted] = useState(false);
  const [selectedIndices, setSelectedIndices] = useState<number[]>([]);

  const isSingleSelect = maxChoices === 1;

  function handleSingleChoose(index: number) {
    if (submitted) return;
    setSubmitted(true);
    send({ type: "SUBMIT_CHOICE", chosen_option_index: index });
    onDone();
  }

  function handleToggle(index: number) {
    if (submitted) return;
    setSelectedIndices((prev) => {
      if (prev.includes(index)) {
        return prev.filter((i) => i !== index);
      }
      if (prev.length >= maxChoices) return prev;
      return [...prev, index];
    });
  }

  function handleConfirmMulti() {
    if (submitted) return;
    if (selectedIndices.length < minChoices || selectedIndices.length > maxChoices) return;
    setSubmitted(true);
    if (selectedIndices.length === 1) {
      send({ type: "SUBMIT_CHOICE", chosen_option_index: selectedIndices[0] });
    } else {
      send({ type: "SUBMIT_CHOICE", chosen_option_indices: selectedIndices });
    }
    onDone();
  }

  if (submitted) {
    return (
      <div className="choice-modal-overlay">
        <div className="choice-modal">
          <p>Waiting...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="choice-modal-overlay">
      <div className="choice-modal">
        <h2>Choose</h2>
        <p>{prompt}</p>
        {!isSingleSelect && (
          <p className="choice-modal__range">
            Select {minChoices === maxChoices ? minChoices : `${minChoices}-${maxChoices}`} option{maxChoices !== 1 ? "s" : ""}
            {" "}&mdash; {selectedIndices.length} selected
          </p>
        )}
        <div className="choice-options">
          {options.map((opt, i) =>
            isSingleSelect ? (
              <button
                key={i}
                className="choice-option"
                onClick={() => handleSingleChoose(i)}
              >
                {opt.label || `Option ${i + 1}`}
              </button>
            ) : (
              <label
                key={i}
                className={`choice-option choice-option--multi ${selectedIndices.includes(i) ? "choice-option--selected" : ""}`}
              >
                <input
                  type="checkbox"
                  checked={selectedIndices.includes(i)}
                  onChange={() => handleToggle(i)}
                  disabled={!selectedIndices.includes(i) && selectedIndices.length >= maxChoices}
                />
                <span>{opt.label || `Option ${i + 1}`}</span>
              </label>
            )
          )}
        </div>
        {!isSingleSelect && (
          <div className="choice-modal__actions">
            <button
              className="choice-confirm"
              disabled={selectedIndices.length < minChoices || selectedIndices.length > maxChoices}
              onClick={handleConfirmMulti}
            >
              Confirm ({selectedIndices.length}/{maxChoices})
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
