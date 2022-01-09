use fallible_collections::TryReserveError;
pub use liq_error::*;
use std::fmt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[allow(non_camel_case_types)]
pub enum liq_error {
    LIQ_OK = 0,
    LIQ_QUALITY_TOO_LOW = 99,
    LIQ_VALUE_OUT_OF_RANGE = 100,
    LIQ_OUT_OF_MEMORY,
    LIQ_ABORTED,
    LIQ_BITMAP_NOT_AVAILABLE,
    LIQ_BUFFER_TOO_SMALL,
    LIQ_INVALID_POINTER,
    LIQ_UNSUPPORTED,
}

impl std::error::Error for liq_error {}

impl fmt::Display for liq_error {
    #[cold]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match *self {
            Self::LIQ_OK => "OK",
            Self::LIQ_QUALITY_TOO_LOW => "QUALITY_TOO_LOW",
            Self::LIQ_VALUE_OUT_OF_RANGE => "VALUE_OUT_OF_RANGE",
            Self::LIQ_OUT_OF_MEMORY => "OUT_OF_MEMORY",
            Self::LIQ_ABORTED => "ABORTED",
            Self::LIQ_BITMAP_NOT_AVAILABLE => "BITMAP_NOT_AVAILABLE",
            Self::LIQ_BUFFER_TOO_SMALL => "BUFFER_TOO_SMALL",
            Self::LIQ_INVALID_POINTER => "INVALID_POINTER",
            Self::LIQ_UNSUPPORTED => "UNSUPPORTED",
        })
    }
}

impl From<TryReserveError> for liq_error {
    #[cold]
    fn from(_: TryReserveError) -> Self {
        Self::LIQ_OUT_OF_MEMORY
    }
}

impl liq_error {
    #[must_use]
    #[inline]
    pub fn is_ok(&self) -> bool {
        *self == liq_error::LIQ_OK
    }

    #[must_use]
    #[inline]
    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }

    #[inline]
    pub fn ok(&self) -> Result<(), liq_error> {
        match *self {
            liq_error::LIQ_OK => Ok(()),
            e => Err(e),
        }
    }

    #[inline]
    pub fn ok_or<E>(self, err: E) -> Result<(), E> {
        if self.is_ok() {
            Ok(())
        } else {
            Err(err)
        }
    }

    pub fn unwrap(&self) {
        assert!(self.is_ok(), "{}", self);
    }

    pub fn expect(&self, msg: &str) {
        assert!(self.is_ok(), "{}", msg);
    }
}
