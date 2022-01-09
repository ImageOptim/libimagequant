use fallible_collections::TryReserveError;
pub use Error::*;
use std::fmt;

#[cfg_attr(feature = "_internal_c_ffi", repr(C))]
#[cfg_attr(not(feature = "_internal_c_ffi"), non_exhaustive)] // it's meant to be always set, but Rust complains for a good but unrelated reason
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[allow(non_camel_case_types)]
pub enum Error {
    #[cfg(feature = "_internal_c_ffi")]
    LIQ_OK = 0,
    QualityTooLow = 99,
    ValueOutOfRange = 100,
    OutOfMemory,
    Aborted,
    BitmapNotAvailable,
    BufferTooSmall,
    InvalidPointer,
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
            Self::BitmapNotAvailable => "BITMAP_NOT_AVAILABLE",
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
