"""
Quantum Support Vector Machine (QSVM) Project
Binary Classification on Iris Dataset

This project implements and compares two different quantum circuit architectures
for binary classification using quantum kernel methods.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

class QuantumCircuitProposal:
    """Base class for quantum circuit proposals"""
    
    def __init__(self, n_qubits: int, name: str):
        self.n_qubits = n_qubits
        self.name = name
        self.dev = qml.device('default.qubit', wires=n_qubits)
        
    def kernel_circuit(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel value between two data points"""
        raise NotImplementedError
        
    def feature_map(self, x: np.ndarray):
        """Quantum feature map to encode data"""
        raise NotImplementedError


class Proposal1_ShallowEntangled(QuantumCircuitProposal):
    """
    Proposal 1: Shallow Circuit with Linear Entanglement
    
    Architecture:
    - 2 layers of parameterized gates
    - Linear entanglement structure (nearest-neighbor CNOT)
    - RY-RZ rotation gates for data encoding
    - Hadamard gates for superposition
    
    Design Rationale:
    - Shallow depth for noise resilience on NISQ devices
    - Linear entanglement reduces gate count while maintaining expressibility
    - RY-RZ gates provide sufficient parameter space
    - Good balance between expressibility and trainability
    """
    
    def __init__(self, n_qubits: int = 2, n_layers: int = 2):
        super().__init__(n_qubits, "Proposal 1: Shallow Linear Entangled")
        self.n_layers = n_layers
        
    def feature_map(self, x: np.ndarray):
        """
        Angle encoding with linear entanglement
        """
        for layer in range(self.n_layers):
            # Hadamard layer for superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Data encoding with RY gates
            for i in range(self.n_qubits):
                qml.RY(x[i % len(x)] * np.pi, wires=i)
            
            # RZ rotation for additional parameters
            for i in range(self.n_qubits):
                qml.RZ(x[i % len(x)] * np.pi / 2, wires=i)
            
            # Linear entanglement (nearest-neighbor)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                
    def kernel_circuit(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum kernel using fidelity"""
        
        @qml.qnode(self.dev)
        def circuit():
            # Encode first data point
            self.feature_map(x1)
            # Apply adjoint of second encoding
            qml.adjoint(self.feature_map)(x2)
            # Return probability of all-zero state
            return qml.probs(wires=range(self.n_qubits))
        
        # Kernel is the probability of measuring |0...0>
        return circuit()[0]


class Proposal2_DeepFullyEntangled(QuantumCircuitProposal):
    """
    Proposal 2: Deep Circuit with Full Entanglement
    
    Architecture:
    - 4 layers of parameterized gates
    - All-to-all entanglement structure
    - RX-RY-RZ rotation gates (full parameter space)
    - CZ gates for entanglement
    
    Design Rationale:
    - Deeper circuit for higher expressibility
    - All-to-all entanglement creates richer quantum correlations
    - Three rotation gates (RX, RY, RZ) span full single-qubit space
    - CZ gates are less prone to noise than CNOT in some hardware
    - Higher expressibility but may face barren plateaus
    """
    
    def __init__(self, n_qubits: int = 2, n_layers: int = 4):
        super().__init__(n_qubits, "Proposal 2: Deep Fully Entangled")
        self.n_layers = n_layers
        
    def feature_map(self, x: np.ndarray):
        """
        Full rotation encoding with all-to-all entanglement
        """
        for layer in range(self.n_layers):
            # Triple rotation for full Bloch sphere coverage
            for i in range(self.n_qubits):
                qml.RX(x[i % len(x)] * np.pi, wires=i)
                qml.RY(x[i % len(x)] * np.pi / 2, wires=i)
                qml.RZ(x[i % len(x)] * np.pi / 4, wires=i)
            
            # All-to-all entanglement using CZ gates
            for i in range(self.n_qubits):
                for j in range(i + 1, self.n_qubits):
                    qml.CZ(wires=[i, j])
                    
    def kernel_circuit(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute quantum kernel using fidelity"""
        
        @qml.qnode(self.dev)
        def circuit():
            self.feature_map(x1)
            qml.adjoint(self.feature_map)(x2)
            return qml.probs(wires=range(self.n_qubits))
        
        return circuit()[0]


class DataPreprocessor:
    """Handle data loading and preprocessing"""
    
    def __init__(self, n_features: int = 2):
        self.n_features = n_features
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_features)
        
    def load_and_preprocess_iris(self) -> Tuple:
        """
        Load Iris dataset and preprocess for binary classification
        
        Preprocessing steps:
        1. Select two classes (Versicolor vs Virginica) for binary classification
        2. Apply PCA for dimensionality reduction to match qubit count
        3. Standardize features using StandardScaler
        
        Rationale:
        - PCA reduces dimensions while preserving maximum variance
        - Standardization ensures features are on similar scales
        - Binary classification simplifies the quantum learning task
        """
        # Load full Iris dataset
        iris = datasets.load_iris()
        X = iris.data # type: ignore
        y = iris.target # type: ignore
        
        # Select two classes for binary classification (1 and 2)
        # Classes: 1=Versicolor, 2=Virginica (these are harder to separate)
        mask = (y == 1) | (y == 2)
        X_binary = X[mask]
        y_binary = y[mask]
        
        # Convert labels to -1 and +1 for SVM
        y_binary = 2 * (y_binary - 1.5)
        
        print(f"Original dataset shape: {X_binary.shape}")
        print(f"Class distribution: {np.unique(y_binary, return_counts=True)}")
        
        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_binary)
        print(f"\nPCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(self.pca.explained_variance_ratio_):.4f}")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X_pca)
        
        # Normalize to [-1, 1] range for quantum encoding
        X_normalized = X_scaled / (np.max(np.abs(X_scaled)) + 1e-8)
        
        print(f"Final dataset shape: {X_normalized.shape}")
        print(f"Feature range: [{X_normalized.min():.3f}, {X_normalized.max():.3f}]")
        
        return X_normalized, y_binary


class ExpressibilityAnalyzer:
    """
    Analyze circuit expressibility using various metrics
    
    Expressibility measures how uniformly a circuit can explore the Hilbert space.
    Higher expressibility means the circuit can represent more diverse quantum states.
    """
    
    def __init__(self, circuit: QuantumCircuitProposal, n_samples: int = 500):
        self.circuit = circuit
        self.n_samples = n_samples
        
    def compute_state_fidelities(self, X_sample: np.ndarray) -> np.ndarray:
        """Compute pairwise state fidelities for random data points"""
        n_points = len(X_sample)
        fidelities = []
        
        # Sample random pairs
        for _ in range(self.n_samples):
            i, j = np.random.choice(n_points, 2, replace=False)
            fidelity = self.circuit.kernel_circuit(X_sample[i], X_sample[j])
            fidelities.append(fidelity)
            
        return np.array(fidelities)
    
    def compute_meyer_wallach_measure(self, X_sample: np.ndarray, n_samples: int = 100) -> float:
        """
        Compute Meyer-Wallach entanglement measure
        
        Q = 2(1 - (1/n) * sum_i Tr(rho_i^2))
        
        where rho_i is the reduced density matrix of qubit i
        Higher Q indicates more entanglement
        """
        dev = self.circuit.dev
        n_qubits = self.circuit.n_qubits
        
        @qml.qnode(dev)
        def get_state(x):
            self.circuit.feature_map(x)
            return qml.state()
        
        Q_values = []
        
        for _ in range(min(n_samples, len(X_sample))):
            idx = np.random.randint(len(X_sample))
            x = X_sample[idx]
            
            # Get full state vector
            state = get_state(x)
            
            # Compute purity for each qubit
            sum_purities = 0.0
            for qubit in range(n_qubits):
                # Reshape state for partial trace
                psi = state.reshape([2] * n_qubits)
                psi = np.moveaxis(psi, qubit, 0).reshape(2, -1)
                
                # Compute reduced density matrix
                rho = psi @ np.conj(psi.T)
                
                # Compute purity
                purity = np.trace(rho @ rho).real
                sum_purities += purity
            
            Q = 2 * (1 - sum_purities / n_qubits)
            Q_values.append(Q)
        
        return np.mean(Q_values)
    
    def analyze_expressibility(self, X_sample: np.ndarray) -> Dict:
        """Comprehensive expressibility analysis"""
        print(f"\nAnalyzing expressibility for {self.circuit.name}...")
        
        # Compute fidelity distribution
        fidelities = self.compute_state_fidelities(X_sample)
        
        # Compute Meyer-Wallach measure
        mw_measure = self.compute_meyer_wallach_measure(X_sample)
        
        # Compute statistics
        results = {
            'mean_fidelity': np.mean(fidelities),
            'std_fidelity': np.std(fidelities),
            'meyer_wallach': mw_measure,
            'fidelity_distribution': fidelities
        }
        
        print(f"  Mean fidelity: {results['mean_fidelity']:.4f}")
        print(f"  Std fidelity: {results['std_fidelity']:.4f}")
        print(f"  Meyer-Wallach measure: {results['meyer_wallach']:.4f}")
        
        return results


class QuantumKernelSVM:
    """Quantum Kernel SVM classifier"""
    
    def __init__(self, circuit: QuantumCircuitProposal):
        self.circuit = circuit
        self.svm = None
        self.kernel_matrix_train = None
        
    def compute_kernel_matrix(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        """Compute kernel matrix between datasets"""
        if X2 is None:
            X2 = X1
            
        n1, n2 = len(X1), len(X2)
        kernel_matrix = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                kernel_matrix[i, j] = self.circuit.kernel_circuit(X1[i], X2[j])
                
        return kernel_matrix
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the quantum kernel SVM"""
        print(f"\nComputing kernel matrix for {self.circuit.name}...")
        self.kernel_matrix_train = self.compute_kernel_matrix(X_train)
        
        # Train classical SVM with precomputed quantum kernel
        self.svm = SVC(kernel='precomputed', C=1.0)
        self.svm.fit(self.kernel_matrix_train, y_train)
        
        print("Training completed!")
        
    def predict(self, X_test: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """Predict labels for test data"""
        kernel_matrix_test = self.compute_kernel_matrix(X_test, X_train)
        return self.svm.predict(kernel_matrix_test)
    
    def score(self, X_test: np.ndarray, y_test: np.ndarray, X_train: np.ndarray) -> float:
        """Compute accuracy score"""
        y_pred = self.predict(X_test, X_train)
        return accuracy_score(y_test, y_pred)


def plot_results(results: Dict, save_path: str = 'results_analysis.png'):
    """Plot comprehensive results comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Quantum SVM Circuit Comparison', fontsize=16, fontweight='bold')
    
    proposals = list(results.keys())
    
    # 1. Accuracy comparison
    ax = axes[0, 0]
    train_accs = [results[p]['train_accuracy'] for p in proposals]
    test_accs = [results[p]['test_accuracy'] for p in proposals]
    x = np.arange(len(proposals))
    width = 0.35
    ax.bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
    ax.bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels([p.split(':')[0] for p in proposals])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Expressibility metrics
    ax = axes[0, 1]
    mw_measures = [results[p]['expressibility']['meyer_wallach'] for p in proposals]
    mean_fids = [results[p]['expressibility']['mean_fidelity'] for p in proposals]
    x = np.arange(len(proposals))
    width = 0.35
    ax.bar(x - width/2, mw_measures, width, label='Meyer-Wallach', alpha=0.8)
    ax.bar(x + width/2, mean_fids, width, label='Mean Fidelity', alpha=0.8)
    ax.set_ylabel('Measure Value')
    ax.set_title('Expressibility Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels([p.split(':')[0] for p in proposals])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Fidelity distribution for Proposal 1
    ax = axes[0, 2]
    fids = results[proposals[0]]['expressibility']['fidelity_distribution']
    ax.hist(fids, bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Fidelity')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{proposals[0].split(":")[0]}\nFidelity Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Fidelity distribution for Proposal 2
    ax = axes[1, 0]
    fids = results[proposals[1]]['expressibility']['fidelity_distribution']
    ax.hist(fids, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xlabel('Fidelity')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{proposals[1].split(":")[0]}\nFidelity Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Circuit depth and gate count comparison
    ax = axes[1, 1]
    metrics = ['Depth', 'Gates', 'Entangling\nGates']
    prop1_vals = [
        results[proposals[0]]['circuit_depth'],
        results[proposals[0]]['gate_count'],
        results[proposals[0]]['entangling_gates']
    ]
    prop2_vals = [
        results[proposals[1]]['circuit_depth'],
        results[proposals[1]]['gate_count'],
        results[proposals[1]]['entangling_gates']
    ]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, prop1_vals, width, label=proposals[0].split(':')[0], alpha=0.8)
    ax.bar(x + width/2, prop2_vals, width, label=proposals[1].split(':')[0], alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title('Circuit Complexity')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 6. Performance summary table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = []
    headers = ['Metric', proposals[0].split(':')[0], proposals[1].split(':')[0]]
    
    table_data.append(['Test Acc.', 
                      f"{results[proposals[0]]['test_accuracy']:.3f}",
                      f"{results[proposals[1]]['test_accuracy']:.3f}"])
    table_data.append(['MW Measure', 
                      f"{results[proposals[0]]['expressibility']['meyer_wallach']:.3f}",
                      f"{results[proposals[1]]['expressibility']['meyer_wallach']:.3f}"])
    table_data.append(['Depth', 
                      f"{results[proposals[0]]['circuit_depth']}",
                      f"{results[proposals[1]]['circuit_depth']}"])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nResults plot saved to {save_path}")
    plt.close()


def get_circuit_specs(circuit: QuantumCircuitProposal) -> Dict:
    """Extract circuit specifications"""
    dev = circuit.dev
    
    @qml.qnode(dev)
    def temp_circuit():
        # Use dummy data
        x = np.array([0.5, 0.5])
        circuit.feature_map(x)
        return qml.probs(wires=0)
    
    # Execute to get specs
    temp_circuit()
    
    # Get circuit depth and gate count
    specs = qml.specs(temp_circuit)()
    
    return {
        'circuit_depth': specs['depth'],
        'gate_count': specs['num_operations'],
        'entangling_gates': specs.get('num_entangling_gates', 0)
    }


def main():
    """Main execution function"""
    print("="*70)
    print("QUANTUM SVM PROJECT: Binary Classification on Iris Dataset")
    print("="*70)
    
    # 1. Load and preprocess data
    print("\n" + "="*70)
    print("STEP 1: Data Loading and Preprocessing")
    print("="*70)
    preprocessor = DataPreprocessor(n_features=2)
    X, y = preprocessor.load_and_preprocess_iris()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\nTrain set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # 2. Initialize circuit proposals
    print("\n" + "="*70)
    print("STEP 2: Initializing Quantum Circuit Proposals")
    print("="*70)
    
    proposal1 = Proposal1_ShallowEntangled(n_qubits=2, n_layers=2)
    proposal2 = Proposal2_DeepFullyEntangled(n_qubits=2, n_layers=4)
    
    circuits = [proposal1, proposal2]
    
    # 3. Analyze expressibility
    print("\n" + "="*70)
    print("STEP 3: Expressibility Analysis")
    print("="*70)
    
    results = {}
    
    for circuit in circuits:
        print(f"\n{'-'*70}")
        print(f"Analyzing: {circuit.name}")
        print(f"{'-'*70}")
        
        # Expressibility analysis
        analyzer = ExpressibilityAnalyzer(circuit, n_samples=300)
        expr_results = analyzer.analyze_expressibility(X_train)
        
        # Get circuit specifications
        circuit_specs = get_circuit_specs(circuit)
        print(f"\nCircuit Specifications:")
        print(f"  Depth: {circuit_specs['circuit_depth']}")
        print(f"  Total gates: {circuit_specs['gate_count']}")
        print(f"  Entangling gates: {circuit_specs['entangling_gates']}")
        
        # Store results
        results[circuit.name] = {
            'expressibility': expr_results,
            **circuit_specs
        }
    
    # 4. Train and evaluate
    print("\n" + "="*70)
    print("STEP 4: Training and Evaluation")
    print("="*70)
    
    for circuit in circuits:
        print(f"\n{'-'*70}")
        print(f"Training: {circuit.name}")
        print(f"{'-'*70}")
        
        # Create and train QSVM
        qsvm = QuantumKernelSVM(circuit)
        qsvm.fit(X_train, y_train)
        
        # Evaluate
        train_acc = qsvm.score(X_train, y_train, X_train)
        test_acc = qsvm.score(X_test, y_test, X_train)
        
        print(f"\nResults:")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
        
        # Get predictions for detailed metrics
        y_pred = qsvm.predict(X_test, X_train)
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Versicolor', 'Virginica']))
        
        # Store results
        results[circuit.name]['train_accuracy'] = train_acc
        results[circuit.name]['test_accuracy'] = test_acc
        results[circuit.name]['y_pred'] = y_pred
    
    # 5. Generate comparison report
    print("\n" + "="*70)
    print("STEP 5: Comparative Analysis")
    print("="*70)
    
    print("\n" + "-"*70)
    print("SUMMARY COMPARISON")
    print("-"*70)
    
    for name, res in results.items():
        print(f"\n{name}")
        print(f"  Test Accuracy: {res['test_accuracy']:.4f}")
        print(f"  Meyer-Wallach Measure: {res['expressibility']['meyer_wallach']:.4f}")
        print(f"  Circuit Depth: {res['circuit_depth']}")
        print(f"  Total Gates: {res['gate_count']}")
        print(f"  Entangling Gates: {res['entangling_gates']}")
    
    # Plot results
    plot_results(results)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()