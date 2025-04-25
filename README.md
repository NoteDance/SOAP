# SOAP

**Overview**:

SOAP (“ShampoO with Adam in the Preconditioner’s eigenbasis”) is motivated by the formal equivalence between Shampoo (with ½ power) and Adafactor run in Shampoo’s eigenbasis. Shampoo delivers superior preconditioning by capturing second-order curvature via Kronecker-factored statistics, but at the cost of extra hyperparameters and compute. SOAP alleviates this by updating the second-moment running average continuously—just like Adam—but performing those updates in Shampoo’s current eigenbasis, avoiding repeated expensive eigendecompositions. The result is an algorithm that matches Shampoo’s superior convergence while approaching Adam’s efficiency, introducing only the “precondition_frequency” hyperparameter beyond Adam’s standard set.

**Parameters**:
- **`learning_rate`** *(float, default=3e-3)*: Base step size for parameter updates.
- **`beta1`** *(float, default=0.95)*: Exponential decay rate for the first-moment (mean) estimate.
- **`beta2`** *(float, default=0.95)*: Exponential decay rate for the second-moment (variance) estimate.
- **`epsilon`** *(float, default=1e-8)*: Small constant for numerical stability in denominator.
- **`weight_decay`** *(float, default=1e-2)*: Coefficient for decoupled weight decay (AdamW style).
- **`shampoo_beta`** *(float or None, default=None)*: Decay rate for Shampoo’s preconditioner; if `None`, uses `beta2`.
- **`precondition_frequency`** *(int, default=10)*: Number of steps between full eigendecomposition updates of the preconditioner.
- **`max_precondition_dim`** *(int, default=10000)*: Maximum dimension for which to apply full matrix preconditioning; larger dims use diagonal fallback.
- **`merge_dims`** *(bool, default=False)*: Whether to collapse small tensor dimensions before preconditioning to reduce cost.
- **`precondition_1d`** *(bool, default=False)*: Enable 1D parameter preconditioning when dimension ≤ `max_precondition_dim`.
- **`correct_bias`** *(bool, default=True)*: Apply bias-correction factors as in Adam.
- **`normalize_gradient`** *(bool, default=False)*: Renormalize gradient magnitude after projection.
- **`data_format`** *(str, default='channels_last')*: Input format for convolutional weights; affects reshape/transpose in preconditioning.
- **`clipnorm`** *(float or None)*: Clip gradients to a maximum L2‐norm.
- **`clipvalue`** *(float or None)*: Clip gradients to a maximum absolute value.
- **`global_clipnorm`** *(float or None)*: Clip all gradients by global norm.
- **`use_ema`** *(bool, default=False)*: Maintain an exponential moving average of weights.
- **`ema_momentum`** *(float, default=0.99)*: Momentum for weight EMA.
- **`ema_overwrite_frequency`** *(int or None)*: Steps between overwriting model weights with EMA weights.
- **`loss_scale_factor`** *(float or None)*: Scale applied to loss for mixed-precision training.
- **`gradient_accumulation_steps`** *(int or None)*: Number of steps to accumulate gradients before applying update.
- **`name`** *(str, default="soap")*: Optional name for the optimizer instance.

---

**Example Usage**:
```python
import tensorflow as tf
from soap import SOAP

# Instantiate the SOAP optimizer
optimizer = SOAP(
    learning_rate=3e-3,
    weight_decay=1e-2,
    precondition_frequency=20,
    merge_dims=True,
)

# Compile a Keras model with SOAP
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=5)
```  
