from common_code.config import get_settings
from common_code.logger.logger import get_logger, Logger
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import FieldDescriptionType, ExecutionUnitTagName, ExecutionUnitTagAcronym
from common_code.common.models import FieldDescription, ExecutionUnitTag
from common_code.tasks.models import TaskData
# Imports required by the service's model
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import io

api_description = """This service re-balances a dataset based on a target class,
it combines oversampling (SMOTE) and under sampling (ENN) to be more generalizable.
In order for the service to work your dataset label column must be called "target".
Finally, avoid having multiple empty lines at the end of the file.
"""
api_summary = """This service re-balances a dataset based on a target class.
"""
api_title = "Class Rebalancing"
version = "0.0.1"

settings = get_settings()


class MyService(Service):
    """
    Rebalance a CSV dataset based on a target column.
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Class Rebalancing",
            slug="class-rebalancing",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="dataset", type=[FieldDescriptionType.TEXT_CSV]),
            ],
            data_out_fields=[
                FieldDescription(
                    name="result", type=[FieldDescriptionType.TEXT_CSV]
                ),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.DATA_PREPROCESSING,
                    acronym=ExecutionUnitTagAcronym.DATA_PREPROCESSING,
                ),
            ],
            has_ai=False,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/class-rebalancing/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        raw = str(data["dataset"].data.decode("utf-8-sig").encode("utf-8"))
        raw = (
            raw.replace(",", ";")
            .replace("\\n", "\n")
            .replace("\\r", "\n")
            .replace("b'", "")
        )

        lines = raw.splitlines()
        if lines[-1] == "" or lines[-1] == "'":
            lines.pop()
        raw = "\n".join(lines)

        df = pd.read_csv(io.StringIO(raw), sep=";")

        # start by removing empty columns that are not target
        df = df.dropna(axis=1, how='all')
        df = df.dropna()

        # create a converter for categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = categorical_columns.to_list() + ['target']
        categorical_columns = {col: pd.Categorical(df[col]) for col in categorical_columns}

        # convert categorical columns to numerical
        for col, cat in categorical_columns.items():
            df[col] = cat.codes

        X, y = df.drop(columns=['target']).to_numpy(), df['target'].to_numpy()

        smote = SMOTE(random_state=42, k_neighbors=1)
        sme = SMOTEENN(random_state=42, smote=smote)
        X_res, y_res = sme.fit_resample(X, y)

        if X_res.shape[0] == 0:
            raise ValueError("SMOTEENN returned an empty dataframe")

        df_res = pd.DataFrame(X_res, columns=df.drop(columns=['target']).columns)
        df_res['target'] = y_res

        # revert the categorical columns
        for col, cat in categorical_columns.items():
            df_res[col] = cat.categories[df_res[col].astype(int)]

        csv_string = df_res.to_csv(index=False)
        csv_bytes = csv_string.encode('utf-8')

        buf = io.BytesIO()
        buf.write(csv_bytes)

        return {
            "result": TaskData(
                data=buf.getvalue(),
                type=FieldDescriptionType.TEXT_CSV
            )
        }
