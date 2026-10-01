#(ly:set-option 'crop #t)
\version "2.24.4"
\header { 
tagline = " "
}
\paper {
    line-width = 150\mm 
}

\drums {
    \numericTimeSignature
    \time 4/4
    <<{
        \partial 8 \repeat unfold 11 hh8 s64
    }
    \\ {
        r8 bd8 r8 sn8 bd8 r8 bd8 sn8 r8 bd8 r8
    }>>
}