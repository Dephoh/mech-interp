interface AccelerateModalProps {
  cardName: string;
  domain: string;
  onConfirm: (payAccelerate: boolean) => void;
}

export function AccelerateModal({ cardName, domain, onConfirm }: AccelerateModalProps) {
  return (
    <div className="accelerate-modal-overlay">
      <div className="accelerate-modal">
        <h2 className="accelerate-modal__title">Accelerate</h2>
        <p className="accelerate-modal__card-name">{cardName}</p>
        <p className="accelerate-modal__desc">
          Pay Accelerate cost? (1E + 1 {domain} Power)
        </p>
        <p className="accelerate-modal__hint">
          Unit will enter ready instead of exhausted
        </p>
        <div className="accelerate-modal__buttons">
          <button
            className="accelerate-modal__btn accelerate-modal__btn--yes"
            onClick={() => onConfirm(true)}
          >
            Accelerate (Enter Ready)
          </button>
          <button
            className="accelerate-modal__btn accelerate-modal__btn--no"
            onClick={() => onConfirm(false)}
          >
            Skip (Enter Exhausted)
          </button>
        </div>
      </div>
    </div>
  );
}
