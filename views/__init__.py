from views import _1_overview as page_1_overview
from views import _2_benchmark as page_2_benchmark
from views import _3_stability as page_3_stability
from views import _4_config as page_4_config
from views import _5_subject as page_5_subject
from views import _6_da as page_6_da
from views import _7_mechanism as page_7_mechanism
from views import _8_target as page_8_target
from views import _9_error as page_9_error
from views import _10_efficiency as page_10_efficiency
try:
    from views import _11_degradation as page_11_degradation
except Exception as _e:
    import logging as _logging
    _logging.error("Failed to import _11_degradation: %s", _e, exc_info=True)
    raise
