"""
Phase 1.2: Structured Information Extraction
Extracts datasets, models, hardware, hyperparameters, and limitations from processed papers
"""

import re
import csv
import json
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractedInfo:
    """Structured information extracted from a paper"""
    paper_id: str
    title: str
    authors: str
    year: Optional[int]
    venue: Optional[str]
    # Data-related
    training_datasets: List[str] = field(default_factory=list)
    training_data_size: Optional[str] = None  # e.g., "970k episodes", "130k demos"
    evaluation_datasets: List[str] = field(default_factory=list)
    # Robot/Hardware
    robot_platforms: List[str] = field(default_factory=list)
    robot_hardware: List[str] = field(default_factory=list)  # sensors, grippers
    compute_hardware: List[str] = field(default_factory=list)  # GPUs, TPUs
    # Model architecture
    model_name: Optional[str] = None  # e.g., "RT-1", "OpenVLA"
    model_architecture: Optional[str] = None  # e.g., "Transformer", "Diffusion Policy"
    model_size: Optional[str] = None  # e.g., "7B", "35M parameters"
    vision_encoder: Optional[str] = None  # e.g., "DINOv2", "EfficientNet"
    base_model: Optional[str] = None  # e.g., "Llama 2", "ViT"
    # Training details
    optimizer: Optional[str] = None
    learning_rate: Optional[str] = None
    batch_size: Optional[str] = None
    epochs: Optional[str] = None
    augmentations: Optional[str] = None
    pretrained_weights: Optional[str] = None
    # Frameworks/Simulation
    simulation_env: Optional[str] = None  # e.g., "Isaac Sim", "MuJoCo"
    ml_framework: Optional[str] = None  # PyTorch, JAX
    # Evaluation
    tasks_evaluated: List[str] = field(default_factory=list)
    success_rate: Optional[str] = None  # overall metric if reported
    baselines_compared: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with lists as comma-separated strings"""
        # Clean up authors field (remove excessive commas/spaces)
        authors_clean = re.sub(r'\s*,\s*,\s*', ', ', self.authors)
        authors_clean = re.sub(r'\s+', ' ', authors_clean).strip()
        
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'authors': authors_clean,
            'year': self.year if self.year else '',
            'venue': self.venue or '',
            'training_datasets': '; '.join(self.training_datasets),  # Use ; instead of ,
            'training_data_size': self.training_data_size or '',
            'evaluation_datasets': '; '.join(self.evaluation_datasets),
            'robot_platforms': '; '.join(self.robot_platforms),
            'robot_hardware': '; '.join(self.robot_hardware),
            'compute_hardware': '; '.join(self.compute_hardware),
            'model_name': self.model_name or '',
            'model_architecture': self.model_architecture or '',
            'model_size': self.model_size or '',
            'vision_encoder': self.vision_encoder or '',
            'base_model': self.base_model or '',
            'optimizer': self.optimizer or '',
            'learning_rate': self.learning_rate or '',
            'batch_size': self.batch_size or '',
            'epochs': self.epochs or '',
            'augmentations': self.augmentations or '',
            'pretrained_weights': self.pretrained_weights or '',
            'simulation_env': self.simulation_env or '',
            'ml_framework': self.ml_framework or '',
            'tasks_evaluated': '; '.join(self.tasks_evaluated[:5]),  # Top 5
            'success_rate': self.success_rate or '',
            'baselines_compared': '; '.join(self.baselines_compared),
        }


class StructuredExtractor:
    """Extract structured information from research paper text."""
    
    def __init__(self):
        # Training/Evaluation Datasets
        self.dataset_patterns = [
            (r'\bOpen[-\s]?X[-\s]?Embodiment\b', 'Open-X Embodiment'),
            (r'\bBridge[-\s]?Data(?:\s+V2)?\b', 'BridgeData V2'),
            (r'\bDROID\b', 'DROID'),
            (r'\bFranka[-\s]?Kitchen\b', 'Franka Kitchen'),
            (r'\bMetaWorld\b', 'MetaWorld'),
            (r'\bRLBench\b', 'RLBench'),
            (r'\bCALVIN\b', 'CALVIN'),
            (r'\bRoboMimic\b', 'RoboMimic'),
            (r'\bLanguage[-\s]?Table\b', 'Language-Table'),
            (r'\bRobo[-\s]?Suite\b', 'RoboSuite'),
            (r'\bManiSkill\b', 'ManiSkill'),
            (r'\bEgo4D\b', 'Ego4D'),
            (r'\bDexCap\b', 'DexCap'),
            (r'\bPH2D\b', 'PH2D'),
            (r'\bSimpler[-\s]?Env\b', 'SimplerEnv'),
        ]
        
        # Robot Platforms
        self.robot_patterns = [
            (r'\bFranka[-\s]?(?:Emika[-\s]?)?Panda\b', 'Franka Panda'),
            (r'\bWidowX\b', 'WidowX'),
            (r'\bGoogle Robot\b', 'Google Robot'),
            (r'\bUnitree[-\s]?H1\b', 'Unitree H1'),
            (r'\bUR5\b', 'UR5'),
            (r'\bUR10\b', 'UR10'),
            (r'\bKinova\b', 'Kinova'),
            (r'\bxArm\b', 'xArm'),
            (r'\bAloha\b', 'Aloha'),
            (r'\bMobile[-\s]?Aloha\b', 'Mobile Aloha'),
        ]
        
        # Compute Hardware (GPUs/TPUs)
        self.compute_patterns = [
            (r'\bA100\b', 'A100'),
            (r'\bA6000\b', 'A6000'),
            (r'\bV100\b', 'V100'),
            (r'\bH100\b', 'H100'),
            (r'\bRTX[-\s]?3090\b', 'RTX 3090'),
            (r'\bRTX[-\s]?4090\b', 'RTX 4090'),
            (r'\bTPU[-\s]?v?\d*\b', 'TPU'),
        ]
        
        # Robot Hardware (sensors, grippers)
        self.robot_hardware_patterns = [
            (r'\bRealSense\b', 'RealSense'),
            (r'\bKinect\b', 'Kinect'),
            (r'\bZED\b', 'ZED'),
            (r'\bInspire\b', 'Inspire Hand'),
            (r'\bRobotiq\b', 'Robotiq Gripper'),
            (r'\bParallel Gripper\b', 'Parallel Gripper'),
        ]
        
        # Vision Encoders
        self.vision_encoder_patterns = [
            (r'\bDINOv2\b', 'DINOv2'),
            (r'\bDINO\b', 'DINO'),
            (r'\bSigLIP\b', 'SigLIP'),
            (r'\bCLIP\b', 'CLIP'),
            (r'\bEfficientNet\b', 'EfficientNet'),
            (r'\bResNet[-\s]?\d*\b', 'ResNet'),
            (r'\bViT\b', 'ViT'),
        ]
        
        # Base Models/Architectures
        self.model_patterns = [
            (r'\bLlama[-\s]?2?\b', 'Llama'),
            (r'\bTransformer\b', 'Transformer'),
            (r'\bDiffusion Policy\b', 'Diffusion Policy'),
            (r'\bACT\b', 'ACT'),
        ]
        
        # VLA/Policy Models (for comparison)
        self.vla_patterns = [
            (r'\bRT[-\s]?1\b', 'RT-1'),
            (r'\bRT[-\s]?2(?:[-\s]?X)?\b', 'RT-2'),
            (r'\bOpenVLA\b', 'OpenVLA'),
            (r'\bOcto\b', 'Octo'),
            (r'\bPaLM[-\s]?E\b', 'PaLM-E'),
            (r'\bRoboCat\b', 'RoboCat'),
            (r'\bGato\b', 'Gato'),
        ]
        
        # Simulation Environments
        self.sim_patterns = [
            (r'\bIsaac[-\s]?Sim\b', 'Isaac Sim'),
            (r'\bIsaac[-\s]?Gym\b', 'Isaac Gym'),
            (r'\bMuJoCo\b', 'MuJoCo'),
            (r'\bPyBullet\b', 'PyBullet'),
            (r'\bSimpler[-\s]?Env\b', 'SimplerEnv'),
            (r'\bHabitat\b', 'Habitat'),
            (r'\bGazebo\b', 'Gazebo'),
        ]
        
        # ML Frameworks
        self.framework_patterns = [
            (r'\bPyTorch\b', 'PyTorch'),
            (r'\bTensorFlow\b', 'TensorFlow'),
            (r'\bJAX\b', 'JAX'),
            (r'\bFlax\b', 'Flax'),
        ]
    
    def extract_from_text(self, text: str, metadata: Dict) -> ExtractedInfo:
        """Extract structured information from paper text."""
        # Clean up authors field - handle cases where it may have partial titles
        authors = metadata.get('authors', 'Unknown')
        if not authors or len(authors.strip()) == 0:
            authors = 'Unknown'
        
        info = ExtractedInfo(
            paper_id=metadata['paper_id'],
            title=metadata['title'],
            authors=authors,
            year=metadata.get('year'),
            venue=metadata.get('venue')
        )
        
        # Extract datasets (both training and evaluation)
        all_datasets = self._extract_patterns(text, self.dataset_patterns)
        info.training_datasets = all_datasets  # Could be refined
        info.evaluation_datasets = all_datasets
        
        # Extract model name from title
        info.model_name = self._extract_model_name(metadata['title'], text)
        
        # Extract data size
        info.training_data_size = self._extract_data_size(text)
        
        # Extract robot platforms
        info.robot_platforms = self._extract_patterns(text, self.robot_patterns)
        
        # Extract hardware
        info.robot_hardware = self._extract_patterns(text, self.robot_hardware_patterns)
        info.compute_hardware = self._extract_patterns(text, self.compute_patterns)
        
        # Extract model architecture info
        info.model_architecture = self._extract_patterns(text, self.model_patterns)
        info.model_architecture = info.model_architecture[0] if info.model_architecture else None
        info.model_size = self._extract_model_size(text)
        info.vision_encoder = self._extract_patterns(text, self.vision_encoder_patterns)
        info.vision_encoder = ', '.join(info.vision_encoder) if info.vision_encoder else None
        
        # Base model
        llm_match = re.search(r'(Llama\s*2?\s*\d+B|GPT[-\s]?\d|T5)', text, re.IGNORECASE)
        info.base_model = llm_match.group(1) if llm_match else None
        
        # Extract training details  
        info.optimizer = self._extract_optimizer(text)
        info.learning_rate = self._extract_learning_rate(text)
        info.batch_size = self._extract_batch_size(text)
        info.epochs = self._extract_epochs(text)
        info.augmentations = self._extract_augmentations(text)
        info.pretrained_weights = self._extract_pretrained(text)
        
        # Extract simulation and framework
        sim_envs = self._extract_patterns(text, self.sim_patterns)
        info.simulation_env = ', '.join(sim_envs) if sim_envs else None
        frameworks = self._extract_patterns(text, self.framework_patterns)
        info.ml_framework = ', '.join(frameworks) if frameworks else None
        
        # Extract evaluation tasks
        info.tasks_evaluated = self._extract_tasks(text)
        
        # Extract success rate
        info.success_rate = self._extract_success_rate(text)
        
        # Extract baselines
        info.baselines_compared = self._extract_patterns(text, self.vla_patterns)
        
        return info
    
    def _extract_patterns(self, text: str, patterns: List[Tuple[str, str]]) -> List[str]:
        """Extract entities matching patterns"""
        found = set()
        for pattern, name in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.add(name)
        return sorted(list(found))
    
    def _extract_model_name(self, title: str, text: str) -> Optional[str]:
        """Extract the model name from title or text"""
        # Common VLA model names
        models = ['RT-1', 'RT-2', 'RT-X', 'OpenVLA', 'Octo', 'ReBot', 'PaLM-E', 
                  'ACT', 'Aloha', 'VIMA', 'RoboCat', 'Gato']
        for model in models:
            if re.search(rf'\b{re.escape(model)}\b', title, re.IGNORECASE):
                return model
        # Check first 1000 chars if not in title
        for model in models:
            if re.search(rf'\b{re.escape(model)}\b', text[:1000], re.IGNORECASE):
                return model
        return None
    
    def _extract_data_size(self, text: str) -> Optional[str]:
        """Extract training data size (e.g., '970k episodes', '130k demos')"""
        patterns = [
            r'(\d+)\s*[kK]\s+(?:real-world\s+)?(?:robot\s+)?(?:manipulation\s+)?(?:episodes|demonstrations?|demos|trajectories)',
            r'(\d+)\s*[mM]\s+(?:real-world\s+)?(?:robot\s+)?(?:manipulation\s+)?(?:episodes|demonstrations?|demos|trajectories)',
            r'(\d+[kKmM])\s+(?:robot\s+)?(?:episodes|demonstrations?|demos|trajectories)',
            r'(\d+,\d+)\s+(?:robot\s+)?(?:episodes|demonstrations?|demos|trajectories)',
            r'(?:trained on|dataset of|consisting of)\s+(\d+)\s*[kKmM]?\s+(?:episodes|demos)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                num = match.group(1)
                # Normalize format: ensure we have k or M suffix
                matched_text = match.group(0).lower()
                if ' k ' in matched_text or 'k ' in matched_text:
                    return f"{num}k episodes"
                elif ' m ' in matched_text or 'm ' in matched_text:
                    return f"{num}M episodes"
                elif 'k' in num.lower() or 'm' in num.lower():
                    return f"{num} episodes"
                else:
                    return f"{num} episodes"
        return None
    
    def _extract_model_size(self, text: str) -> Optional[str]:
        """Extract model size (e.g., '7B', '35M parameters')"""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*B[-\s]parameter',  # "7 B-parameter" or "7B-parameter"
            r'(\d+(?:\.\d+)?)\s*M[-\s]parameter',  # "35 M-parameter" or "35M-parameter"
            r'(\d+)\s*[Bb]illion\s+parameters?',    # "7 billion parameters"
            r'(\d+)\s*[Mm]illion\s+parameters?',    # "35 million parameters"
            r'(\d+(?:\.\d+)?)\s*B\s+(?:model|parameters?)',  # "7B model" or "7B parameters"
            r'(\d+(?:\.\d+)?)\s*M\s+(?:model|parameters?)',  # "35M model" or "35M parameters"
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE)
            if match:
                size = match.group(1)
                # Add suffix based on pattern
                if 'billion' in match.group(0).lower() or ' B' in match.group(0) or 'B-' in match.group(0):
                    return size + 'B'
                elif 'million' in match.group(0).lower() or ' M' in match.group(0) or 'M-' in match.group(0):
                    return size + 'M'
        return None
    
    def _extract_tasks(self, text: str) -> List[str]:
        """Extract number of evaluation tasks"""
        # Look for "X tasks" or "X different tasks" mentions
        patterns = [
            r'(\d+)\s+(?:evaluation\s+)?tasks',
            r'evaluated\s+on\s+(\d+)\s+tasks',
            r'(\d+)\s+manipulation\s+tasks',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                return [f"{match.group(1)} tasks"]
        
        # Look for specific task names in evaluation section
        eval_section = text[text.lower().find('evaluation'):text.lower().find('evaluation')+3000] if 'evaluation' in text.lower() else ''
        if eval_section:
            # Find quoted task names like "pick apple", "place cup"
            quoted_tasks = re.findall(r'"([^"]{10,50})"', eval_section)
            if quoted_tasks:
                # Clean and limit to reasonable task descriptions
                clean_tasks = [t.strip() for t in quoted_tasks if 10 < len(t) < 50]
                return clean_tasks[:5]  # Top 5
        
        return []
    
    def _extract_success_rate(self, text: str) -> Optional[str]:
        """Extract overall success rate if mentioned"""
        patterns = [
            r'by\s+([\d.]+)%\s+(?:absolute\s+)?success',  # "by 16.5% absolute success"
            r'([\d.]+)%\s+(?:absolute\s+)?success\s+rate',  # "16.5% absolute success rate"
            r'success\s+rate[s]?\s+of\s+([\d.]+)%',  # "success rate of 16.5%"
            r'achiev(?:es?|ing)\s+(?:at\s+least\s+)?([\d.]+)%',  # "achieving 90%"
            r'([\d.]+)%\s+success',  # "16.5% success"
        ]
        # Look in abstract and results sections
        abstract_end = min(2000, len(text))
        results_start = text.lower().find('results')
        search_text = text[:abstract_end]
        if results_start > 0:
            search_text += text[results_start:results_start+3000]
        
        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                rate = match.group(1)
                # Only return if it's a reasonable success rate (0-100)
                try:
                    if 0 <= float(rate) <= 100:
                        return rate + '%'
                except ValueError:
                    pass
        return None
    
    def _extract_learning_rate(self, text: str) -> Optional[str]:
        """Extract learning rate"""
        patterns = [
            r'learning\s+rate(?:\s+of)?\s*[=:]\s*([\d.e\-]+)',
            r'\blr\s*[=:]\s*([\d.e\-]+)',
            r'(?:initial|base)\s+learning\s+rate[:\s]+([\d.e\-]+)',
            r'with\s+a\s+learning\s+rate\s+of\s+([\d.e\-]+)',
        ]
        
        # Search in methods/training section first
        train_section = text.lower().find('training')
        search_text = text[train_section:train_section+5000] if train_section > 0 else text[:10000]
        
        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_optimizer(self, text: str) -> Optional[str]:
        """Extract optimizer"""
        # Search in training/methods section
        train_section = text.lower().find('training')
        search_text = text[train_section:train_section+5000] if train_section > 0 else text[:10000]
        
        patterns = [
            r'\b(AdamW)\b',  # Check AdamW first (more specific)
            r'\b(Adam)\b',
            r'\b(SGD)\b',
            r'\b(RMSprop)\b',
            r'\b(LAMB)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_batch_size(self, text: str) -> Optional[str]:
        """Extract batch size"""
        # Search in training section
        train_section = text.lower().find('training')
        search_text = text[train_section:train_section+5000] if train_section > 0 else text[:10000]
        
        patterns = [
            r'batch\s+size(?:\s+of)?\s*[=:]\s*(\d+)',
            r'mini[-\s]?batch\s+size[:\s]*(\d+)',
            r'with\s+(?:a\s+)?batch\s+size\s+of\s+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_epochs(self, text: str) -> Optional[str]:
        """Extract epochs"""
        train_section = text.lower().find('training')
        search_text = text[train_section:train_section+5000] if train_section > 0 else text[:10000]
        
        patterns = [
            r'(?:for|trained\s+for|over)\s+(\d+)\s+epochs',
            r'epochs\s*[=:]\s*(\d+)',
            r'(\d+)\s+training\s+epochs',
        ]
        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_augmentations(self, text: str) -> Optional[str]:
        """Extract data augmentation techniques"""
        aug_keywords = [
            'random crop', 'center crop', 'flip', 'horizontal flip', 
            'rotation', 'color jitter', 'random erase', 'mixup', 'cutout',
            'random shift', 'augmentation'
        ]
        found = []
        search_text = text[:15000]  # Search more text
        for keyword in aug_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', search_text, re.IGNORECASE):
                found.append(keyword.title())
        return ', '.join(found[:5]) if found else None
    
    def _extract_pretrained(self, text: str) -> Optional[str]:
        """Extract pretrained weights info"""
        patterns = [
            r'pretrained\s+on\s+(ImageNet|COCO|[\w\s-]+?)(?:\s|,|\.|;)',
            r'initialized\s+(?:from|with)\s+([\w\s-]+?)(?:\s+weights|\s+model)',
            r'fine-tun(?:e|ed|ing)\s+([\w\s-]+?)(?:\s+model|\s+weights)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                result = match.group(1).strip()
                # Clean up common artifacts
                result = re.sub(r'\s+', ' ', result)
                if len(result) < 50:  # Reasonable length
                    return result
        return None
    
    def batch_extract(self, processed_papers_dir: str, output_csv: str = "extracted_info.csv") -> List[ExtractedInfo]:
        """
        Extract information from all processed papers and save to CSV.
        
        Args:
            processed_papers_dir: Directory containing processed papers
            output_csv: Output CSV file path
            
        Returns:
            List of ExtractedInfo objects
        """
        papers_path = Path(processed_papers_dir)
        
        # Load metadata
        metadata_file = papers_path / "metadata.json"
        if not metadata_file.exists():
            print(f"Error: {metadata_file} not found")
            return []
        
        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)
        
        # Process each paper
        fulltext_dir = papers_path / "full_text"
        all_info = []
        
        print(f"Extracting structured information from {len(metadata_list)} papers...")
        
        for i, meta_dict in enumerate(metadata_list, 1):
            paper_id = meta_dict['paper_id']
            print(f"[{i}/{len(metadata_list)}] Processing: {paper_id}")
            
            # Read full text
            text_file = fulltext_dir / f"{paper_id}.txt"
            if not text_file.exists():
                print(f"  ✗ Full text not found: {text_file}")
                continue
            
            with open(text_file, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            # Extract information
            info = self.extract_from_text(full_text, meta_dict)
            all_info.append(info)
            
            print(f"  ✓ Model: {info.model_name or 'N/A'} ({info.model_size or 'N/A'})")
            print(f"  ✓ Datasets: {', '.join(info.training_datasets[:3]) if info.training_datasets else 'None'}")
            print(f"  ✓ Robots: {', '.join(info.robot_platforms) if info.robot_platforms else 'None'}")
        
        # Save to CSV with proper quoting
        output_path = Path(output_csv)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'paper_id', 'title', 'authors', 'year', 'venue',
                'training_datasets', 'training_data_size', 'evaluation_datasets',
                'robot_platforms', 'robot_hardware', 'compute_hardware',
                'model_name', 'model_architecture', 'model_size', 
                'vision_encoder', 'base_model',
                'optimizer', 'learning_rate', 'batch_size', 'epochs', 
                'augmentations', 'pretrained_weights',
                'simulation_env', 'ml_framework',
                'tasks_evaluated', 'success_rate', 'baselines_compared'
            ], quoting=csv.QUOTE_ALL)  # Quote all fields to handle commas/special chars
            writer.writeheader()
            for info in all_info:
                writer.writerow(info.to_dict())
        
        print(f"\n{'='*60}")
        print(f"✓ Extraction complete!")
        print(f"✓ Processed {len(all_info)} papers")
        print(f"✓ CSV saved to: {output_path}")
        
        # Print summary statistics
        total_datasets = sum(len(info.training_datasets) for info in all_info)
        total_robots = sum(len(info.robot_platforms) for info in all_info)
        papers_with_size = sum(1 for info in all_info if info.model_size)
        print(f"\n📊 Summary:")
        print(f"  - Total datasets mentioned: {total_datasets}")
        print(f"  - Total robot platforms: {total_robots}")
        print(f"  - Papers with model size: {papers_with_size}")
        print(f"  - Papers with success rates: {sum(1 for info in all_info if info.success_rate)}")
        
        return all_info


if __name__ == "__main__":
    import sys
    
    print(f"Phase 1.2: Structured Information Extraction")
    print(f"{'='*60}\n")
    
    processed_dir = sys.argv[1] if len(sys.argv) > 1 else "./processed_papers"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "extracted_info.csv"
    
    extractor = StructuredExtractor()
    extractor.batch_extract(processed_dir, output_csv)

