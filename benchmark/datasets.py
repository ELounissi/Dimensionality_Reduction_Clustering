from __future__ import annotations
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .config import (
    REAL_FIGSHARE_ARTICLE,
    RSG_FIGSHARE_ARTICLE,
    SPECTF_URL,
    STUDENT_FIGSHARE_ARTICLES,
    stable_seed,
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    display_name: str
    family: str
    scope: str
    n_clusters: int
    n_features: int
    replicate: int
    source: str
    path: str = ""


REAL_SPECS = (
    ("breast_tissue", "Breast tissue", "breast_tissue", 6, 9),
    ("breast_wisconsin", "Breast Wisconsin", "wdbc", 2, 30),
    ("ecoli", "Ecoli", "ecoli", 8, 7),
    ("glass", "Glass", "glass", 6, 9),
    ("haberman", "Haberman", "haberman", 2, 3),
    ("ionosphere", "Ionosphere", "ionosphere", 2, 34),
    ("iris", "Iris", "iris", 3, 4),
    ("movement_libras", "Movement libras", "movement_libras", 15, 90),
    ("musk", "Musk", "musk", 2, 166),
    ("parkinsons", "Parkinsons", "parkinsons", 2, 22),
    ("segmentation", "Segmentation", "segmentation", 7, 19),
    ("sonar", "Sonar all", "sonar", 2, 60),
    ("spectf", "Spectf", "spectf", 2, 44),
    ("transfusion", "Transfusion", "transfusion", 2, 4),
    ("vehicle", "Vehicle", "vehicle", 4, 18),
    ("vertebral_column", "Vertebral column", "vertebral", 3, 6),
    ("vowel_context", "Vowel context", "vowel_context", 11, 10),
    ("wine", "Wine", "wine", 3, 13),
    ("wine_quality_red", "Wine quality red", "wine_quality_red", 6, 11),
    ("yeast", "Yeast", "yeast", 10, 8),
)
REAL_SAMPLE_COUNTS = {
    "breast_tissue": 106,
    "breast_wisconsin": 569,
    "ecoli": 336,
    "glass": 214,
    "haberman": 306,
    "ionosphere": 351,
    "iris": 150,
    "movement_libras": 360,
    "musk": 476,
    "parkinsons": 195,
    "segmentation": 2310,
    "sonar": 208,
    "spectf": 267,
    "transfusion": 748,
    "vehicle": 846,
    "vertebral_column": 310,
    "vowel_context": 990,
    "wine": 178,
    "wine_quality_red": 1599,
    "yeast": 1484,
}
REQUIRED_REAL_FILES = {
    "BreastTissue.xls",
    "wdbc.data",
    "ecoli.data",
    "glass.data",
    "haberman.data",
    "ionosphere.data",
    "iris.data",
    "movement_libras.data",
    "clean1.data.Z",
    "parkinsons.data",
    "segmentation.data",
    "segmentation.test",
    "sonar.all-data",
    "transfusion.data",
    "xaa.dat",
    "xab.dat",
    "xac.dat",
    "xad.dat",
    "xae.dat",
    "xaf.dat",
    "xag.dat",
    "xah.dat",
    "xai.dat",
    "column_3C.dat",
    "vowel-context.data",
    "wine.data",
    "winequality-red.csv",
    "yeast.data",
}
RSG_PATTERN = re.compile(
    "C(?P<clusters>\\d+)F(?P<features>\\d+)N(?P<distributions>\\d+)Ne(?P<per_cluster>\\d+).*R(?P<replicate>\\d+)\\.(?:txt|arff)$"
)
CIRCLES_MOONS_PATTERN = re.compile(
    "make_(?P<family>circles|moons)(?:_C(?P<k5>5))?_dim(?P<features>\\d+)_run(?P<run>\\d+)\\.csv$"
)
REPLICLUST_PATTERN = re.compile(
    "D(?P<features>\\d+)_K(?P<clusters>\\d+)_run(?P<run>\\d+)\\.csv$"
)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_md5: str | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not expected_md5 or _md5(destination) == expected_md5):
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    with (
        urllib.request.urlopen(url, timeout=300) as response,
        temporary.open("wb") as output,
    ):
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if expected_md5 and _md5(temporary) != expected_md5:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {destination.name}")
    temporary.replace(destination)


def _article(article_id: int) -> dict:
    records = json.loads(
        (Path(__file__).resolve().parents[1] / "assets/sources.json").read_text(
            encoding="utf-8"
        )
    )
    return next(record for record in records if record["id"] == article_id)


def _extract_rar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.which("bsdtar"):
        subprocess.run(
            ["bsdtar", "-xf", str(archive), "-C", str(destination)], check=True
        )
    elif shutil.which("unrar"):
        subprocess.run(
            ["unrar", "x", "-o+", str(archive), str(destination) + "/"], check=True
        )
    elif shutil.which("7z"):
        subprocess.run(["7z", "x", "-y", f"-o{destination}", str(archive)], check=True)
    else:
        raise RuntimeError(
            "extracting the RAR archives requires bsdtar (libarchive), unrar or 7z"
        )


def fetch_data(data_root: Path, include_rsg: bool = True) -> None:
    raw = data_root / "raw"
    real_dir = raw / "real"
    real = _article(REAL_FIGSHARE_ARTICLE)
    real_dir.mkdir(parents=True, exist_ok=True)
    (real_dir / "figshare_article.json").write_text(
        json.dumps(real, indent=2) + "\n", encoding="utf-8"
    )
    for item in real["files"]:
        if item["name"] in REQUIRED_REAL_FILES:
            _download(
                item["download_url"], real_dir / item["name"], item.get("supplied_md5")
            )
    spect_zip = real_dir / "spectf_heart.zip"
    _download(SPECTF_URL, spect_zip)
    spect_dir = real_dir / "spectf_heart"
    spect_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(spect_zip) as archive:
        archive.extractall(spect_dir)
    student_dir = raw / "student"
    student_dir.mkdir(parents=True, exist_ok=True)
    for family, article_id in STUDENT_FIGSHARE_ARTICLES.items():
        article = _article(article_id)
        (student_dir / f"article_{article_id}.json").write_text(
            json.dumps(article, indent=2) + "\n", encoding="utf-8"
        )
        item = article["files"][0]
        archive = student_dir / item["name"]
        _download(item["download_url"], archive, item.get("supplied_md5"))
        target = student_dir / archive.stem
        if not any(target.rglob("*.csv")):
            _extract_rar(archive, target)
        if len(list(target.rglob("*.csv"))) != 300:
            raise ValueError(f"expected 300 {family} datasets under {target}")
    if include_rsg:
        rsg = _article(RSG_FIGSHARE_ARTICLE)
        rsg_dir = raw / "rsg"
        rsg_dir.mkdir(parents=True, exist_ok=True)
        (rsg_dir / "figshare_article.json").write_text(
            json.dumps(rsg, indent=2) + "\n", encoding="utf-8"
        )
        folders = rsg.get("folder_structure", {})
        for item in rsg["files"]:
            relative = Path(folders.get(str(item["id"]), "")) / item["name"]
            _download(
                item["download_url"], rsg_dir / relative, item.get("supplied_md5")
            )
    files = [
        path
        for path in sorted(raw.rglob("*"))
        if path.is_file() and path.suffix != ".part"
    ]
    manifest = {
        "sources": {
            "real_figshare_article": REAL_FIGSHARE_ARTICLE,
            "student_figshare_articles": STUDENT_FIGSHARE_ARTICLES,
            "rsg_figshare_article": RSG_FIGSHARE_ARTICLE if include_rsg else None,
            "spectf_url": SPECTF_URL,
        },
        "files": [
            {
                "path": path.relative_to(data_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
            if path.suffix != ".csv" or "student" not in path.parts
        ],
    }
    reference = json.loads(
        (Path(__file__).resolve().parents[1] / "assets/data_checksums.json").read_text()
    )
    for record in reference["files"]:
        path = data_root / record["path"]
        if path.exists() and _sha256(path) != record["sha256"]:
            raise ValueError(f"Input SHA-256 mismatch: {record['path']}")
    (data_root / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def student_specs(data_root: Path) -> list[DatasetSpec]:
    root = data_root / "raw" / "student"
    specs: list[DatasetSpec] = []
    for path in sorted(root.rglob("*.csv")):
        match = CIRCLES_MOONS_PATTERN.match(path.name)
        if match:
            family = match["family"]
            clusters = 5 if match["k5"] else 2
            features = int(match["features"])
            replicate = int(match["run"]) - 1
        else:
            match = REPLICLUST_PATTERN.match(path.name)
            if not match:
                continue
            family = "repliclust"
            clusters = int(match["clusters"])
            features = int(match["features"])
            replicate = int(match["run"])
        dataset_id = f"{family}_k{clusters}_d{features}_r{replicate:02d}"
        specs.append(
            DatasetSpec(
                dataset_id,
                dataset_id,
                family,
                "synthetic",
                clusters,
                features,
                replicate,
                f"figshare:{STUDENT_FIGSHARE_ARTICLES[family]}",
                path.relative_to(data_root).as_posix(),
            )
        )
    if not specs:
        raise FileNotFoundError(
            f"no archived Circles/Moons/Repliclust datasets under {root}; run fetch-data"
        )
    return sorted(specs, key=lambda spec: spec.dataset_id)


def rsg_specs(data_root: Path) -> list[DatasetSpec]:
    root = data_root / "raw" / "rsg"
    specs: list[DatasetSpec] = []
    for path in sorted([*root.rglob("*.txt"), *root.rglob("*.arff")]):
        match = RSG_PATTERN.match(path.name)
        if not match:
            continue
        values = {key: int(value) for key, value in match.groupdict().items()}
        dataset_id = f"rsg_{path.stem}"
        specs.append(
            DatasetSpec(
                dataset_id,
                path.stem,
                "rsg",
                "synthetic",
                values["clusters"],
                values["features"],
                values["replicate"],
                f"figshare:{RSG_FIGSHARE_ARTICLE}",
                path.relative_to(data_root).as_posix(),
            )
        )
    if not specs:
        raise FileNotFoundError(f"no RSG files found under {root}; run fetch-data")
    return specs


def real_specs() -> list[DatasetSpec]:
    selected = REAL_SPECS
    return [
        DatasetSpec(
            name,
            display,
            "real",
            "real",
            clusters,
            features,
            0,
            f"loader:{loader}",
            loader,
        )
        for name, display, loader, clusters, features in selected
    ]


def dataset_specs(data_root: Path) -> list[DatasetSpec]:
    return [*student_specs(data_root), *rsg_specs(data_root), *real_specs()]


def expected_samples(spec: DatasetSpec) -> int:
    if spec.scope == "real":
        return REAL_SAMPLE_COUNTS[spec.dataset_id]
    if spec.family == "rsg":
        match = RSG_PATTERN.match(Path(spec.path).name)
        return int(match["clusters"]) * int(match["per_cluster"]) if match else 0
    return 2000


def _numeric_rows(path: Path, expected_columns: int) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        for row in csv.reader(stream):
            if len(row) != expected_columns:
                continue
            try:
                [float(value) for value in row[1:]]
            except ValueError:
                continue
            rows.append(row)
    return rows


def _load_real(loader: str, root: Path) -> tuple[np.ndarray, np.ndarray]:
    if loader == "breast_tissue":
        frame = pd.read_excel(root / "BreastTissue.xls", sheet_name="Data")
        y = frame["Class"].astype(str).to_numpy()
        x = frame.drop(
            columns=[
                column for column in frame.columns if column in {"Class", "Case #"}
            ]
        ).to_numpy()
    elif loader == "wdbc":
        frame = pd.read_csv(root / "wdbc.data", header=None)
        y, x = (frame.iloc[:, 1].to_numpy(), frame.iloc[:, 2:].to_numpy())
    elif loader == "ecoli":
        frame = pd.read_csv(root / "ecoli.data", sep="\\s+", header=None)
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, 1:-1].to_numpy())
    elif loader == "glass":
        frame = pd.read_csv(root / "glass.data", header=None)
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, 1:-1].to_numpy())
    elif loader == "haberman":
        frame = pd.read_csv(root / "haberman.data", header=None)
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, :-1].to_numpy())
    elif loader == "ionosphere":
        frame = pd.read_csv(root / "ionosphere.data", header=None)
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, :-1].to_numpy())
    elif loader == "iris":
        frame = pd.read_csv(root / "iris.data", header=None).dropna(how="all")
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, :-1].to_numpy())
    elif loader == "movement_libras":
        frame = pd.read_csv(root / "movement_libras.data", header=None)
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, :-1].to_numpy())
    elif loader == "musk":
        from unlzw3 import unlzw

        decoded = unlzw((root / "clean1.data.Z").read_bytes()).decode("utf-8")
        frame = pd.read_csv(io.StringIO(decoded), header=None)
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, 2:-1].to_numpy())
    elif loader == "parkinsons":
        frame = pd.read_csv(root / "parkinsons.data")
        y, x = (
            frame["status"].to_numpy(),
            frame.drop(columns=["name", "status"]).to_numpy(),
        )
    elif loader == "segmentation":
        rows = [
            *_numeric_rows(root / "segmentation.data", 20),
            *_numeric_rows(root / "segmentation.test", 20),
        ]
        frame = pd.DataFrame(rows)
        y, x = (frame.iloc[:, 0].to_numpy(), frame.iloc[:, 1:].astype(float).to_numpy())
    elif loader == "sonar":
        frame = pd.read_csv(root / "sonar.all-data", header=None)
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, :-1].to_numpy())
    elif loader == "spectf":
        files = list((root / "spectf_heart").rglob("SPECTF.train")) + list(
            (root / "spectf_heart").rglob("SPECTF.test")
        )
        if len(files) != 2:
            raise FileNotFoundError(
                "expected SPECTF.train and SPECTF.test from UCI dataset 96"
            )
        frame = pd.concat(
            [pd.read_csv(path, header=None) for path in sorted(files)],
            ignore_index=True,
        )
        y, x = (frame.iloc[:, 0].to_numpy(), frame.iloc[:, 1:].to_numpy())
    elif loader == "transfusion":
        frame = pd.read_csv(root / "transfusion.data")
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, :-1].to_numpy())
    elif loader == "vehicle":
        frame = pd.concat(
            [
                pd.read_csv(path, sep="\\s+", header=None)
                for path in sorted(root.glob("x*.dat"))
            ],
            ignore_index=True,
        )
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, :-1].to_numpy())
    elif loader == "vertebral":
        frame = pd.read_csv(root / "column_3C.dat", sep="\\s+", header=None)
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, :-1].to_numpy())
    elif loader == "vowel_context":
        frame = pd.read_csv(root / "vowel-context.data", sep="\\s+", header=None)
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, 3:13].to_numpy())
    elif loader == "wine":
        frame = pd.read_csv(root / "wine.data", header=None)
        y, x = (frame.iloc[:, 0].to_numpy(), frame.iloc[:, 1:].to_numpy())
    elif loader == "wine_quality_red":
        frame = pd.read_csv(root / "winequality-red.csv", sep=";")
        y, x = (frame["quality"].to_numpy(), frame.drop(columns="quality").to_numpy())
    elif loader == "yeast":
        frame = pd.read_csv(root / "yeast.data", sep="\\s+", header=None)
        y, x = (frame.iloc[:, -1].to_numpy(), frame.iloc[:, 1:-1].to_numpy())
    else:
        raise ValueError(f"unknown real-data loader: {loader}")
    return (np.asarray(x, dtype=np.float64), np.asarray(y))


def load_dataset(spec: DatasetSpec, data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    if spec.family in ("circles", "moons", "repliclust"):
        frame = pd.read_csv(data_root / spec.path)
        x, y = (
            frame.iloc[:, :-1].to_numpy(dtype=np.float64),
            frame.iloc[:, -1].to_numpy(),
        )
    elif spec.family == "rsg":
        if Path(spec.path).suffix.lower() == ".arff":
            with (data_root / spec.path).open(
                encoding="utf-8", errors="strict"
            ) as stream:
                data_line = next(
                    (
                        index
                        for index, line in enumerate(stream)
                        if line.strip().lower() == "@data"
                    )
                )
            frame = np.loadtxt(
                data_root / spec.path, delimiter=",", skiprows=data_line + 1
            )
            x, y = (frame[:, :-1], frame[:, -1])
        else:
            frame = np.loadtxt(data_root / spec.path)
            x, y = (frame[:, :-1], frame[:, -1])
    elif spec.scope == "real":
        x, y = _load_real(spec.path, data_root / "raw" / "real")
    else:
        raise ValueError(f"unknown dataset family: {spec.family}")
    x = StandardScaler().fit_transform(np.asarray(x, dtype=np.float64))
    y = pd.factorize(np.asarray(y), sort=True)[0]
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError(
            f"invalid shapes for {spec.dataset_id}: X={x.shape}, y={y.shape}"
        )
    if x.shape[1] != spec.n_features:
        raise ValueError(
            f"{spec.dataset_id}: expected {spec.n_features} features, got {x.shape[1]}"
        )
    if len(np.unique(y)) != spec.n_clusters:
        raise ValueError(
            f"{spec.dataset_id}: expected {spec.n_clusters} clusters, got {len(np.unique(y))}"
        )
    if not np.isfinite(x).all():
        raise ValueError(f"non-finite values in {spec.dataset_id}")
    return (x, y)


def dataset_manifest(specs: list[DatasetSpec], data_root: Path) -> pd.DataFrame:
    rows = []
    for spec in specs:
        x, y = load_dataset(spec, data_root)
        rows.append(
            {
                **asdict(spec),
                "n_samples": len(x),
                "observed_features": x.shape[1],
                "observed_clusters": len(np.unique(y)),
                "minimum_cluster_size": int(np.bincount(y).min()),
                "maximum_cluster_size": int(np.bincount(y).max()),
                "seed": stable_seed(spec.dataset_id),
            }
        )
    return pd.DataFrame(rows)
