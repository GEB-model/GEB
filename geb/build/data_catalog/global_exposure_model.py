"""Utilities to fetch and convert Global Exposure Model (GEM) CSV exposures.

This module provides an adapter that downloads CSV files from the
`gem/global_exposure_model` GitHub repository for one or more countries,
processes those CSVs to compute damages per square metre for residential
(`OCCUPANCY == 'Res'`) assets and writes the aggregated parameters to a
JSON file named `global_exposure_model.json` next to the adapter's path.

Operations performed by the adapter:
- Check for existing processed CSV files to avoid redundant downloads.
- Query the GitHub tree API to find CSV file paths for requested countries.
- Download matching raw CSV files from the `raw.githubusercontent.com` URL.
- Parse CSVs with pandas, filter residential rows and aggregate by
    `NAME_1` (admin_1) to compute damage per sqm for several damage columns.


The class is intentionally lightweight: network interactions use
`requests`, temporary files are used for pandas reads, and numerical
division is guarded against zero area values.
"""

import os
import tempfile
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from .base import Adapter

# A mapping to correct specific GADM names that may appear in the CSVs, to
# ensure consistency with expected admin_1 region names. This is a simple
# hardcoded mapping based on observed discrepancies; it can be extended as
# needed if more mismatches are found. The keys are the incorrect names as
# they appear in the CSVs, and the values are the corrected names that should
# be used in the output (matching gadm v2.8).
gadm_converter: dict[str, str] = {
    "QuichÃ©": "Quiche",
    "TotonicapÃ¡n": "Totonicapan",
    "Veracruz de Ignacio de la Llave": "Veracruz",
    "Michoacán de Ocampo": "Michoacan",
    "Ciudad de México": "Distrito Federal",
    "PetÃ©n": "Peten",
    "SololÃ¡": "Solola",
    "Tyrol": "Tirol",
    "Vaduz": "Valduz",
    "Luzern": "Lucerne",
    "St. Gallen": "Sankt Gallen",
    "Balochistan": "Baluchistan",
    # Russia: oblasts whose base name differs from GADM (apostrophes, transliteration)
    # Simple "X Oblast" → "X" cases are handled automatically by canon() stripping
    # " Oblast"; only entries where the stripped name still doesn't match GADM are listed.
    "Arkhangelsk Oblast": "Arkhangel'sk",
    "Astrakhan Oblast": "Astrakhan'",
    "Murmansk Oblast": "Murmansk",
    "Nizhny Novgorod Oblast": "Nizhegorod",
    "Novgorod Oblast": "Novgorod",
    "Oryol Oblast": "Orel",
    "Pskov Oblast": "Pskov",
    "Ryazan Oblast": "Ryazan'",
    "Smolensk Oblast": "Smolensk",
    "Tver Oblast": "Tver'",
    "Tyumen Oblast": "Tyumen'",
    "Ulyanovsk Oblast": "Ul'yanovsk",
    "Yaroslavl Oblast": "Yaroslavl'",
    # Russia: Krais (GADM uses shorter or transliterated names)
    "Altai Krai": "Altay",
    "Kamchatka Krai": "Kamchatka",
    "Khabarovsk Krai": "Khabarovsk",
    "Krasnodar Krai": "Krasnodar",
    "Krasnoyarsk Krai": "Krasnoyarsk",
    "Perm Krai": "Perm'",
    "Primorsky Krai": "Primor'ye",
    "Stavropol Krai": "Stavropol'",
    "Zabaykalsky Krai": "Zabaykal'ye",
    # Russia: Republics (GEM uses full official names; GADM uses shortened forms)
    "Altai Republic": "Gorno-Altay",  # Republic of Altai ≠ Altai Krai in GADM
    "Komi Republic": "Komi",
    "Republic of Karelia": "Karelia",
    "Republic of Mordovia": "Mordovia",
    "Sakha Republic": "Sakha",
    # Russia: autonomous okrugs / oblasts (GADM uses abbreviated names)
    "Chukotka Autonomous Okrug": "Chukot",
    "Jewish Autonomous Oblast": "Yevrey",
    "Khanty-Mansiysk Autonomous Okrug - Ugra": "Khanty-Mansiy",
    "Nenets Autonomous Okrug": "Nenets",
    "Yamalo-Nenets Autonomous Okrug": "Yamal-Nenets",
    # Russia: republics where GEM drops the "Republic" suffix but GADM uses a
    # completely different transliteration
    "Adygea": "Adygey",
    "Buryatia": "Buryat",
    "Chuvashia": "Chuvash",
    "Ingushetia": "Ingush",
    "Kabardino-Balkaria": "Kabardin-Balkar",
    "Kalmykia": "Kalmyk",
    "Karachay-Cherkessia": "Karachay-Cherkess",
    "Khakassia": "Khakass",
    "Mari El": "Mariy-El",
    "North Ossetia-Alania": "North Ossetia",
    "Udmurtia": "Udmurt",
    # Russia: federal cities and Moscow Oblast
    "Saint Petersburg": "City of St. Petersburg",
    "Moscow": "Moscow City",
    "Moscow Oblast": "Moskva",
    # Russia: Magadan Oblast has a quirky name in GADM v2.8
    "Magadan Oblast": "Maga Buryatdan",
    # Turkey: GEM uses ALL-CAPS Turkish names; canon() normalises case, but
    # one province has a genuinely different GADM name (GADM v2.8 uses the
    # old short name "Afyon" instead of the full official name). The key uses
    # the raw GEM CSV spelling (with Turkish capital İ) to match exactly.
    "AFYONKARAHİSAR": "Afyon",
    # Georgia: GEM uses English translations or "Autonomous Republic of ..."
    # prefixes that differ from GADM v2.8 transliterations.
    "Autonomous Republic of Abkhazia": "Abkhazia",
    "Autonomous Republic of Adjara": "Ajaria",
    "Lower Kartli": "Kvemo Kartli",  # kvemo = lower in Georgian
    "Inner Kartli": "Shida Kartli",  # shida = inner in Georgian
    "Samegrelo-Upper Svaneti": "Samegrelo-Zemo Svaneti",  # zemo = upper
    "Racha-Lechkhumi and Lower Svaneti": "Racha-Lechkhumi-Kvemo Svaneti",
    # Austria: GEM uses English names; GADM v2.8 uses German names.
    "Carinthia": "Kärnten",
    "Lower Austria": "Niederösterreich",
    "Upper Austria": "Oberösterreich",
    "Styria": "Steiermark",
    "Vienna": "Wien",
    # Italy: GEM uses Italian names that differ from GADM v2.8 equivalents.
    "Puglia": "Apulia",
    "Sicilia": "Sicily",
    # Mexico: GEM uses the full official state name; GADM uses the short form.
    "Coahuila de Zaragoza": "Coahuila",
}

# Mapping from GADM country names (NAME_0, underscores replacing spaces) to
# the folder names used in the GEM GitHub repository, where they differ.
gem_country_name_aliases: dict[str, str] = {
    "Czech_Republic": "Czechia",
    # Åland Islands is an autonomous Finnish archipelago with no GEM folder;
    # Finland is used as a proxy.
    "Aland": "Finland",
    # Faroe Islands is a Danish autonomous territory with no GEM folder;
    # Denmark is used as a proxy.
    "Faroe_Islands": "Denmark",
    # UK Crown Dependencies: Guernsey and Jersey have no GEM folder;
    # United Kingdom is used as a proxy.
    "Guernsey": "United_Kingdom",
    "Jersey": "United_Kingdom",
    # Akrotiri and Dhekelia is a UK Sovereign Base Area on Cyprus; Cyprus is
    # used as a proxy because it shares the same physical and construction context.
    "Akrotiri_and_Dhekelia": "Cyprus",
    # Northern Cyprus is a de-facto state on the island of Cyprus with no GEM
    # folder; Cyprus is used as a proxy.
    "Northern_Cyprus": "Cyprus",
    # Greenland is a Danish autonomous territory with no GEM folder;
    # Denmark is used as a proxy.
    "Greenland": "Denmark",
    # GADM uses the Spanish name "Palestina"; GEM uses "Palestine".
    "Palestina": "Palestine",
    # Western Sahara is a disputed territory with no GEM folder;
    # Morocco administers most of it and is used as a proxy.
    "Western_Sahara": "Morocco",
    # San Marino is a microstate enclave within Italy with no GEM folder;
    # Italy is used as a proxy.
    "San_Marino": "Italy",
    # Svalbard and Jan Mayen is a Norwegian territory (largely uninhabited)
    # with no GEM folder; Norway is used as a proxy.
    "Svalbard_and_Jan_Mayen": "Norway",
    # Vatican City / Holy See is a microstate enclave within Italy with no
    # GEM folder; Italy is used as a proxy. GADM may list this as either name.
    "Vatican": "Italy",
    "Holy_See": "Italy",
    # GADM uses the former name "Macedonia"; GEM uses the current official
    # name "North_Macedonia" (since 2019).
    "Macedonia": "North_Macedonia",
}


class GlobalExposureModel(Adapter):
    """Adapter for Global Exposure Model data."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the GlobalExposureModel adapter.

        Args:
            *args: Positional arguments to pass to the Adapter constructor.
            **kwargs: Keyword arguments to pass to the Adapter constructor.
        """
        super().__init__(*args, **kwargs)

    def canon(self, string_to_normalize: str) -> str:
        """Canonicalizes a string by normalizing it to ASCII and stripping whitespace.

        Some characters (e.g. Polish Ł/ł) do not decompose to ASCII via NFKD
        and would be silently dropped, causing mismatches (e.g. "Łódź" → "odz").
        These are mapped to their closest ASCII equivalents first.

        Args:
            string_to_normalize: The string to canonicalize.
        Returns:
            The canonicalized string.
        """
        # Characters that NFKD cannot decompose to ASCII must be substituted
        # explicitly; otherwise encode("ascii", "ignore") drops them entirely.
        # æ/Æ are ligatures that must be expanded to ae/AE before NFKD; they
        # cannot be handled by str.maketrans (one-to-one only).
        s = string_to_normalize.replace("æ", "ae").replace("Æ", "Ae")
        # ı (U+0131, Turkish dotless i) and Ł/ł/Ø/ø/Ð/ð have no NFKD
        # decomposition to ASCII; map them explicitly before normalisation.
        _non_decomposable = str.maketrans("ŁłØøÐð\u0131", "LlOoDdi")
        s = s.translate(_non_decomposable)
        s = (
            unicodedata.normalize("NFKD", s)
            .encode("ascii", "ignore")
            .decode("ascii")
            .strip()
        )
        # Normalise ALL-CAPS strings (e.g. GEM Turkey uses "ADANA", GADM uses
        # "Adana") so that both sides produce the same canon key.
        if s.isupper():
            s = s.title()
        # Strip the trailing " oblast" suffix (case-insensitive) so that GEM
        # names like "Leningrad Oblast" and GADM names like "Leningrad" both
        # resolve to the same key without manually enumerating every oblast.
        if s.lower().endswith(" oblast"):
            s = s[: -len(" oblast")]
        return s

    def _filter_folders(
        self, tree: list[dict[str, Any]], csv_files: list[str], country: str
    ) -> None:
        """Filter the repository tree for folders matching the specified country.

        Args:
            tree: The repository tree from the GitHub API.
            csv_files: List to append found CSV file paths to.
            country: The country name to filter folders by.
        Raises:
            ValueError: If no matching country folder is found. Add an entry to
                ``gem_country_name_aliases`` to map the country to a proxy if GEM
                does not cover it.
        """
        matching_paths = [
            item["path"]
            for item in tree
            if item["type"] == "tree"
            and item["path"].split("/")[-1].lower() == country.lower()
        ]

        if not matching_paths:
            raise ValueError(
                f"No folder found for country '{country}' in the GEM repository. "
                f"Add an entry to gem_country_name_aliases in "
                f"geb/build/data_catalog/global_exposure_model.py to map it to a "
                f"proxy country, or verify the country name against the GEM repository."
            )

        for item in tree:
            if item["type"] == "blob":
                for folder in matching_paths:
                    if (
                        item["path"].startswith(folder + "/")
                        and "exposure_res" in item["path"].lower()
                    ):
                        csv_files.append(item["path"])

    def _download_and_process_csv(
        self, raw_base: str, csv_files: list[str], countries_to_download: list[str]
    ) -> None:
        """Download and process CSV files from the repository.

        Args:
            raw_base: The base URL for raw file access in the GitHub repository.
            csv_files: List of CSV file paths to download and process.
            countries_to_download: List of country names corresponding to the CSV files.
        Raises:
            ValueError: If an expected column is missing in the CSV or if processing fails.
        """
        # Will collect per-file dictionaries mapping admin_1 -> damage metrics
        # assert the lengths of csv_files and countries_to_download match to ensure correct pairing
        if len(csv_files) != len(countries_to_download):
            raise ValueError(
                "The number of CSV files and countries to download must match. "
                f"Got {len(csv_files)} CSV files and {len(countries_to_download)} countries."
            )
        for csv, country in zip(csv_files, countries_to_download):
            url = raw_base + quote(csv)

            r = requests.get(url)
            r.raise_for_status()

            with tempfile.TemporaryDirectory() as temp_dir_str:
                temp_dir: Path = Path(temp_dir_str)
                out_path = temp_dir / os.path.basename(csv)
                with open(out_path, "wb") as f:
                    f.write(r.content)
                # Read the downloaded CSV into a pandas DataFrame for
                # processing. Using a temporary file avoids holding raw
                # bytes in memory and keeps pandas happy with local paths.
                df = pd.read_csv(out_path)
                if "NAME_1" not in df.columns:
                    # Fail explicitly with a clear error if the expected admin_1
                    # column is missing, rather than raising an opaque KeyError
                    # later during processing.
                    raise ValueError(
                        "Expected column 'NAME_1' to be present in GEM exposure CSV "
                        f"'{csv}', but it was not found."
                    )
                df_processed = pd.DataFrame(self._process_csv(df)).T
                df_processed.index.name = "NAME_1"
                # write to path
                fn_country_data = (
                    self.path.parent / f"global_exposure_model_{country}.csv"
                )
                df_processed.to_csv(fn_country_data, index=True)

    def _process_csv(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Process a single CSV DataFrame to compute damages per square meter.

        Args:
            df: The DataFrame to process.
        Returns:
            A dictionary with damages per square meter by admin_1 region.
        """
        # Only consider residential occupancy rows for damage-per-sqm
        # calculations.
        result = {}
        df = df[df["OCCUPANCY"] == "Res"]
        for admin_1 in df["NAME_1"].dropna().unique():
            result[admin_1] = {}
            for damage_type in [
                "TOTAL_REPL_COST_USD",
                "COST_STRUCTURAL_USD",
                "COST_NONSTRUCTURAL_USD",
                "COST_CONTENTS_USD",
            ]:
                df_admin_1 = df[df["NAME_1"] == admin_1]
                total_damage = df_admin_1[damage_type].sum()
                total_area = df_admin_1["TOTAL_AREA_SQM"].sum()
                # Compute damage per square metre, guarding against
                # division-by-zero when total_area is zero or missing.
                result[admin_1][damage_type + "_SQM"] = float(
                    total_damage / total_area if total_area > 0 else 0
                )
        return result

    def fetch(self, url: str, countries: list[str]) -> GlobalExposureModel:
        """Fetch and process data for specific countries.

        Args:
            url: required but not used in this implementation.
            countries: The list of countries to fetch data for.
        Returns:
            GlobalExposureModel: The adapter instance with the processed data.
        """
        # set attribute of the adapter to the list of countries for which data is being fetched
        countries = [country.replace(" ", "_") for country in countries]
        # Apply country-name aliases upfront so fetch() and read() both use the
        # same (aliased) name when constructing cached CSV filenames.
        self.countries = [
            gem_country_name_aliases.get(self.canon(c), self.canon(c))
            for c in countries
        ]

        # Query the repository tree for all files in the `main` branch so we
        # can locate country-specific folders without cloning the repo.
        branch = "main"
        tree_url = f"https://api.github.com/repos/gem/global_exposure_model/git/trees/{branch}?recursive=1"
        resp = requests.get(tree_url)  # , headers=HEADERS)
        resp.raise_for_status()
        tree = resp.json()["tree"]
        # Base URL for fetching raw file contents by path
        raw_base = (
            f"https://raw.githubusercontent.com/gem/global_exposure_model/{branch}/"
        )

        csv_files_to_download = []
        countries_to_download = []
        for country in self.countries:
            # Normalize the country name to ASCII for matching against
            # repository folder names (avoids accented-character mismatches).
            country = (
                unicodedata.normalize("NFKD", country)
                .encode("ascii", "ignore")
                .decode("ascii")
            )
            # replace spaces with dashes for matching folder names and search
            country = country.replace(" ", "_")
            # apply any known country-name aliases (e.g. GADM "Czech_Republic" → GEM "Czechia")
            country = gem_country_name_aliases.get(country, country)
            # create pathname and check if exist in GEB datacatalog
            fn_country_data = self.path.parent / f"global_exposure_model_{country}.csv"
            if not fn_country_data.exists():
                self._filter_folders(tree, csv_files_to_download, country)
                countries_to_download.append(country)
        if not csv_files_to_download:
            return self  # No new files to download, return early
        self._download_and_process_csv(
            raw_base, csv_files_to_download, countries_to_download
        )

        return self

    def read(self, **kwargs: Any) -> dict[str, dict[str, float]]:
        """Read the dataset into a dictionary.

        Args:
            **kwargs: Additional keyword arguments to pass to the reader function.
        Returns:
            The dataset as a dictionary.
        """
        # read all country-specific CSV files that were downloaded and processed, and merge them into a single dictionary
        merged_result: dict[str, dict[str, float]] = {}
        for country in self.countries:
            fn_country_data = self.path.parent / f"global_exposure_model_{country}.csv"
            df = pd.read_csv(fn_country_data)
            name_1_series = df["NAME_1"]
            # check if any of the converted GADM names need to be applied
            if any(key in name_1_series.values for key in gadm_converter.keys()):
                df["NAME_1"] = name_1_series.replace(gadm_converter)
            # merge the processed data for this country into the overall result dictionary
            for _, row in df.iterrows():
                admin_1 = self.canon(row["NAME_1"])
                if admin_1 not in merged_result:
                    merged_result[admin_1] = {}
                for col in df.columns:
                    if col != "NAME_1":
                        merged_result[admin_1][col] = row[col]
            # Store a national average under a private key so that aliased
            # territories (e.g. Faroe Islands NAME_1 regions when Denmark is
            # used as a proxy) can fall back to it in setup_building_reconstruction_costs.
            numeric_cols = [c for c in df.columns if c != "NAME_1"]
            if numeric_cols:
                country_avg = df[numeric_cols].mean(numeric_only=True).to_dict()
                merged_result[f"_country_avg_{country}"] = country_avg
        return merged_result
