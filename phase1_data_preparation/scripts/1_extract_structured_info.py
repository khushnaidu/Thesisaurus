import re
import csv
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractedInfo:
    paper_id: str
    title: str
    year: Optional[int]
    venue: Optional[str]
    training_datasets: List[str] = field(default_factory=list)
    training_data_size: Optional[str] = None
    evaluation_datasets: List[str] = field(default_factory=list)
    robot_platforms: List[str] = field(default_factory=list)
    robot_hardware: List[str] = field(default_factory=list)
    compute_hardware: List[str] = field(default_factory=list)
    model_name: Optional[str] = None
    model_architecture: Optional[str] = None
    model_size: Optional[str] = None
    vision_encoder: Optional[str] = None
    base_model: Optional[str] = None
    optimizer: Optional[str] = None
    learning_rate: Optional[str] = None
    batch_size: Optional[str] = None
    epochs: Optional[str] = None
    augmentations: Optional[str] = None
    pretrained_weights: Optional[str] = None
    simulation_env: Optional[str] = None
    ml_framework: Optional[str] = None
    tasks_evaluated: List[str] = field(default_factory=list)
    success_rate: Optional[str] = None
    baselines_compared: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'year': self.year if self.year else '',
            'venue': self.venue or '',
            'training_datasets': '; '.join(self.training_datasets),
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
            'tasks_evaluated': '; '.join(self.tasks_evaluated[:5]),
            'success_rate': self.success_rate or '',
            'baselines_compared': '; '.join(self.baselines_compared),
        }


class StructuredExtractor:
    def __init__(self):
        # dataset patterns
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

        # robot patterns
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

        # compute hw
        self.compute_patterns = [
            (r'\bA100\b', 'A100'),
            (r'\bA6000\b', 'A6000'),
            (r'\bV100\b', 'V100'),
            (r'\bH100\b', 'H100'),
            (r'\bRTX[-\s]?3090\b', 'RTX 3090'),
            (r'\bRTX[-\s]?4090\b', 'RTX 4090'),
            (r'\bTPU[-\s]?v?\d*\b', 'TPU'),
        ]

        # robot hw (sensors etc)
        self.robot_hardware_patterns = [
            (r'\bRealSense\b', 'RealSense'),
            (r'\bKinect\b', 'Kinect'),
            (r'\bZED\b', 'ZED'),
            (r'\bInspire\b', 'Inspire Hand'),
            (r'\bRobotiq\b', 'Robotiq Gripper'),
            (r'\bParallel Gripper\b', 'Parallel Gripper'),
        ]

        # vision encoders
        self.vision_encoder_patterns = [
            (r'\bDINOv2\b', 'DINOv2'),
            (r'\bDINO\b', 'DINO'),
            (r'\bSigLIP\b', 'SigLIP'),
            (r'\bCLIP\b', 'CLIP'),
            (r'\bEfficientNet\b', 'EfficientNet'),
            (r'\bResNet[-\s]?\d*\b', 'ResNet'),
            (r'\bViT\b', 'ViT'),
        ]

        # model architectures
        self.model_patterns = [
            (r'\bLlama[-\s]?2?\b', 'Llama'),
            (r'\bTransformer\b', 'Transformer'),
            (r'\bDiffusion Policy\b', 'Diffusion Policy'),
            (r'\bACT\b', 'ACT'),
        ]

        # vla models for comparison
        self.vla_patterns = [
            (r'\bRT[-\s]?1\b', 'RT-1'),
            (r'\bRT[-\s]?2(?:[-\s]?X)?\b', 'RT-2'),
            (r'\bOpenVLA\b', 'OpenVLA'),
            (r'\bOcto\b', 'Octo'),
            (r'\bPaLM[-\s]?E\b', 'PaLM-E'),
            (r'\bRoboCat\b', 'RoboCat'),
            (r'\bGato\b', 'Gato'),
        ]

        # sim envs
        self.sim_patterns = [
            (r'\bIsaac[-\s]?Sim\b', 'Isaac Sim'),
            (r'\bIsaac[-\s]?Gym\b', 'Isaac Gym'),
            (r'\bMuJoCo\b', 'MuJoCo'),
            (r'\bPyBullet\b', 'PyBullet'),
            (r'\bSimpler[-\s]?Env\b', 'SimplerEnv'),
            (r'\bHabitat\b', 'Habitat'),
            (r'\bGazebo\b', 'Gazebo'),
        ]

        # ml frameworks
        self.framework_patterns = [
            (r'\bPyTorch\b', 'PyTorch'),
            (r'\bTensorFlow\b', 'TensorFlow'),
            (r'\bJAX\b', 'JAX'),
            (r'\bFlax\b', 'Flax'),
        ]

    def extract_from_text(self, text: str, metadata: Dict) -> ExtractedInfo:
        info = ExtractedInfo(
            paper_id=metadata['paper_id'],
            title=metadata['title'],
            year=metadata.get('year'),
            venue=metadata.get('venue')
        )

        all_datasets = self._extract_patterns(text, self.dataset_patterns)
        info.training_datasets = all_datasets
        info.evaluation_datasets = all_datasets

        info.model_name = self._extract_model_name(metadata['title'], text)
        info.training_data_size = self._extract_data_size(text)
        info.model_size = self._extract_model_size(text)

        info.robot_platforms = self._extract_patterns(text, self.robot_patterns)
        info.robot_hardware = self._extract_patterns(text, self.robot_hardware_patterns)
        info.compute_hardware = self._extract_patterns(text, self.compute_patterns)

        info.model_architecture = self._extract_patterns(text, self.model_patterns)
        info.model_architecture = info.model_architecture[0] if info.model_architecture else None
        info.vision_encoder = self._extract_patterns(text, self.vision_encoder_patterns)
        info.vision_encoder = ', '.join(info.vision_encoder) if info.vision_encoder else None

        llm_match = re.search(r'(Llama\s*2?\s*\d+B|GPT[-\s]?\d|T5)', text, re.IGNORECASE)
        info.base_model = llm_match.group(1) if llm_match else None

        info.optimizer = self._extract_optimizer(text)
        info.learning_rate = self._extract_learning_rate(text)
        info.batch_size = self._extract_batch_size(text)
        info.epochs = self._extract_epochs(text)
        info.augmentations = self._extract_augmentations(text)
        info.pretrained_weights = self._extract_pretrained(text)

        sim_envs = self._extract_patterns(text, self.sim_patterns)
        info.simulation_env = ', '.join(sim_envs) if sim_envs else None
        frameworks = self._extract_patterns(text, self.framework_patterns)
        info.ml_framework = ', '.join(frameworks) if frameworks else None

        info.tasks_evaluated = self._extract_tasks(text)
        info.success_rate = self._extract_success_rate(text)
        info.baselines_compared = self._extract_patterns(text, self.vla_patterns)

        return info

    def _extract_patterns(self, text: str, patterns: List[Tuple[str, str]]) -> List[str]:
        found = set()
        for pattern, name in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.add(name)
        return sorted(list(found))

    def _extract_model_name(self, title: str, text: str) -> Optional[str]:
        models = ['RT-1', 'RT-2', 'RT-X', 'OpenVLA', 'Octo', 'ReBot', 'PaLM-E',
                  'ACT', 'Aloha', 'VIMA', 'RoboCat', 'Gato']
        for model in models:
            if re.search(rf'\b{re.escape(model)}\b', title, re.IGNORECASE):
                return model
        for model in models:
            if re.search(rf'\b{re.escape(model)}\b', text[:1000], re.IGNORECASE):
                return model
        return None

    def _extract_data_size(self, text: str) -> Optional[str]:
        patterns = [
            r'(\d+)\s*[kK]\s+(?:real-world\s+)?(?:robot\s+)?(?:manipulation\s+)?(?:episodes|demonstrations?|demos|trajectories)',
            r'(\d+)\s*[mM]\s+(?:real-world\s+)?(?:robot\s+)?(?:manipulation\s+)?(?:episodes|demonstrations?|demos|trajectories)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                num = match.group(1)
                matched_text = match.group(0).lower()
                if 'k ' in matched_text or 'k' in num.lower():
                    return f"{num}k episodes"
                elif 'm ' in matched_text or 'm' in num.lower():
                    return f"{num}M episodes"
        return None

    def _extract_model_size(self, text: str) -> Optional[str]:
        patterns = [
            r'(\d+(?:\.\d+)?)\s*B[-\s]parameter',
            r'(\d+(?:\.\d+)?)\s*[Bb]illion\s+parameters?',
            r'(\d+(?:\.\d+)?)\s*B\s+(?:model|parameters?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:3000], re.IGNORECASE)
            if match:
                return match.group(1) + 'B'
        return None

    def _extract_tasks(self, text: str) -> List[str]:
        patterns = [
            r'(\d+)\s+(?:evaluation\s+)?tasks',
            r'evaluated\s+on\s+(\d+)\s+tasks',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:5000], re.IGNORECASE)
            if match:
                return [f"{match.group(1)} tasks"]
        return []

    def _extract_success_rate(self, text: str) -> Optional[str]:
        patterns = [
            r'success\s+rate[s]?\s+of\s+([\d.]+)%',
            r'achiev(?:es?|ing)\s+(?:at\s+least\s+)?([\d.]+)%',
        ]
        search_text = text[:2000]
        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                rate = match.group(1)
                try:
                    if 0 <= float(rate) <= 100:
                        return rate + '%'
                except ValueError:
                    pass
        return None

    def _extract_learning_rate(self, text: str) -> Optional[str]:
        patterns = [
            r'learning\s+rate(?:\s+of)?\s*[=:]\s*([\d.e\-]+)',
            r'\blr\s*[=:]\s*([\d.e\-]+)',
        ]
        train_section = text.lower().find('training')
        search_text = text[train_section:train_section+5000] if train_section > 0 else text[:10000]
        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_optimizer(self, text: str) -> Optional[str]:
        patterns = [r'\b(AdamW)\b', r'\b(Adam)\b', r'\b(SGD)\b', r'\b(RMSprop)\b']
        train_section = text.lower().find('training')
        search_text = text[train_section:train_section+5000] if train_section > 0 else text[:10000]
        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_batch_size(self, text: str) -> Optional[str]:
        pattern = r'batch\s+size(?:\s+of)?\s*[=:]\s*(\d+)'
        train_section = text.lower().find('training')
        search_text = text[train_section:train_section+5000] if train_section > 0 else text[:10000]
        match = re.search(pattern, search_text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_epochs(self, text: str) -> Optional[str]:
        pattern = r'(?:for|trained\s+for|over)\s+(\d+)\s+epochs'
        train_section = text.lower().find('training')
        search_text = text[train_section:train_section+5000] if train_section > 0 else text[:10000]
        match = re.search(pattern, search_text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_augmentations(self, text: str) -> Optional[str]:
        aug_keywords = ['random crop', 'center crop', 'flip', 'rotation', 'color jitter']
        found = []
        for keyword in aug_keywords:
            if re.search(rf'\b{re.escape(keyword)}\b', text[:15000], re.IGNORECASE):
                found.append(keyword.title())
        return ', '.join(found[:5]) if found else None

    def _extract_pretrained(self, text: str) -> Optional[str]:
        pattern = r'pretrained\s+on\s+(ImageNet|COCO|[\w\s-]+?)(?:\s|,|\.|;)'
        match = re.search(pattern, text[:5000], re.IGNORECASE)
        if match:
            result = match.group(1).strip()
            if len(result) < 50:
                return result
        return None

    def batch_extract(self, data_dir: str, output_csv: str) -> List[ExtractedInfo]:
        data_path = Path(data_dir)

        metadata_file = data_path / "metadata.json"
        if not metadata_file.exists():
            print(f"error: {metadata_file} not found")
            return []

        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)

        fulltext_dir = data_path / "full_text"
        all_info = []

        print(f"extracting info from {len(metadata_list)} papers...")

        for i, meta_dict in enumerate(metadata_list, 1):
            paper_id = meta_dict['paper_id']
            print(f"[{i}/{len(metadata_list)}] {paper_id}")

            text_file = fulltext_dir / f"{paper_id}.txt"
            if not text_file.exists():
                print(f"  - full text not found")
                continue

            with open(text_file, 'r', encoding='utf-8') as f:
                full_text = f.read()

            info = self.extract_from_text(full_text, meta_dict)
            all_info.append(info)

            print(f"  model: {info.model_name or 'n/a'}")
            print(f"  datasets: {', '.join(info.training_datasets[:3]) if info.training_datasets else 'none'}")

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'paper_id', 'title', 'year', 'venue',
                'training_datasets', 'training_data_size', 'evaluation_datasets',
                'robot_platforms', 'robot_hardware', 'compute_hardware',
                'model_name', 'model_architecture', 'model_size',
                'vision_encoder', 'base_model',
                'optimizer', 'learning_rate', 'batch_size', 'epochs',
                'augmentations', 'pretrained_weights',
                'simulation_env', 'ml_framework',
                'tasks_evaluated', 'success_rate', 'baselines_compared'
            ], quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for info in all_info:
                writer.writerow(info.to_dict())

        print(f"\n{'='*50}")
        print(f"done! processed {len(all_info)} papers")
        print(f"csv saved to: {output_path}")
        print('='*50)

        return all_info


if __name__ == "__main__":
    import sys

    print("phase 1.2: structured info extraction")
    print("="*50 + "\n")

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "../outputs/extracted_info.csv"

    extractor = StructuredExtractor()
    extractor.batch_extract(data_dir, output_csv)
