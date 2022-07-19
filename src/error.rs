use std::collections::TryReserveError;
use std::fmt;
pub use Error::*;

/// Error codes
#[cfg_attr(feature = "_internal_c_ffi", repr(C))]
#[cfg_attr(not(feature = "_internal_c_ffi"), non_exhaustive)] // it's meant to be always set, but Rust complains for a good but unrelated reason
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[allow(non_camel_case_types)]
pub enum Error {
    /// Not an error. Exists for back-compat with the C API
    #[cfg(feature = "_internal_c_ffi")]
    LIQ_OK = 0,
    /// [`set_quality()`][crate::Attributes::set_quality] was used with a minimum quality, and the minimum could not be achieved
    QualityTooLow = 99,
    /// Function called with invalid arguments
    ValueOutOfRange = 100,
    /// Either the system/process really hit a limit, or some data like image size was ridiculously wrong. Could be a bug too
    OutOfMemory,
    /// Progress callback said to stop
    Aborted,
    /// Some terrible inconsistency happened
    InternalError,
    /// Slice needs to be bigger, or width/height needs to be smaller
    BufferTooSmall,
    /// NULL pointer or use-after-free in the C API
    InvalidPointer,
    /// Congratulations, you've discovered an edge case
    Unsupported,
}

impl std::error::Error for Error {}

impl fmt::Display for Error {
    #[cold]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            #[cfg(feature = "_internal_c_ffi")]
            Self::LIQ_OK => "OK",
            Self::QualityTooLow => "QUALITY_TOO_LOW",
            Self::ValueOutOfRange => "VALUE_OUT_OF_RANGE",
            Self::OutOfMemory => "OUT_OF_MEMORY",
            Self::Aborted => "ABORTED",
            Self::InternalError => "INTERNAL_ERROR",
            Self::BufferTooSmall => "BUFFER_TOO_SMALL",
            Self::InvalidPointer => "INVALID_POINTER",
            Self::Unsupported => "UNSUPPORTED",
        })
    }
}

impl From<TryReserveError> for Error {
    #[cold]
    fn from(_: TryReserveError) -> Self {
        Self::OutOfMemory
    }
}
