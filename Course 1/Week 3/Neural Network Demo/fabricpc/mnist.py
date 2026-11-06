import jax
from fabricpc.nodes import Linear, IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import TanhActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer
import optax
from fabricpc.training import train, evaluate
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc import setup_jax

setup_jax()  # auto-detects the backend; pass platform="cuda" to require GPU
jax.config.update("jax_default_prng_impl", "threefry2x32")

# --- Step 1: Define Nodes ---
pixels = IdentityNode(shape=(784,), name="pixels")
hidden1 = Linear(
    shape=(256,),
    activation=TanhActivation(),
    name="hidden1",
    weight_init=XavierInitializer(distribution="normal", gain=1.0),
)
hidden2 = Linear(
    shape=(64,),
    activation=TanhActivation(),
    name="hidden2",
    weight_init=XavierInitializer(distribution="normal", gain=1.0),
)
output = Linear(
    shape=(10,),
    activation=SoftmaxActivation(),
    energy=CrossEntropyEnergy(),  # Use cross-entropy for classification
    name="class",
    weight_init=XavierInitializer(distribution="normal", gain=1.0),
)

# --- Step 2: Build Graph ---
structure = graph(
    nodes=[pixels, hidden1, hidden2, output],
    edges=[
        Edge(source=pixels, target=hidden1.slot("in")),
        Edge(source=hidden1, target=hidden2.slot("in")),
        Edge(source=hidden2, target=output.slot("in")),
    ],
    task_map=TaskMap(x=pixels, y=output),  # x=input, y=target
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)

# --- Step 3: Initialize Parameters ---
# Matching seed values do not produce identical NumPy/JAX weights or shuffles.
master_rng_key = jax.random.PRNGKey(3)
graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)
params = initialize_params(structure, graph_key)

# --- Step 4: Train ---
train_config = {"num_epochs": 20}
batch_size = 128
optimizer = optax.sgd(learning_rate=0.1)

train_loader = MnistLoader(
    "train", batch_size=batch_size, tensor_format="flat", shuffle=True, seed=3,
    normalize_mean=0.0, normalize_std=1.0,
)
test_loader = MnistLoader(
    "test", batch_size=batch_size, tensor_format="flat", shuffle=False,
    normalize_mean=0.0, normalize_std=1.0,
)

result = train(
    params=params,
    structure=structure,
    train_loader=train_loader,
    optimizer=optimizer,
    config=train_config,
    rng_key=train_key,
    verbose=True,
)
trained_params = result.params

# --- Step 5: Evaluate with a forward pass, matching NumPy's predict() ---
# This option only selects evaluation behavior; training above still uses PC.
metrics = evaluate(
    trained_params, structure, test_loader, train_config, eval_key,
    algorithm="backprop",
)
print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")

