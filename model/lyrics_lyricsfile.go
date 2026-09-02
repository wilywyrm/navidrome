package model

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/navidrome/navidrome/utils/str"
	"gopkg.in/yaml.v3"
)

// parseLyricsfile parses a LRCLIB Lyricsfile YAML document
// (see https://github.com/tranxuanthang/lrcget/blob/main/LYRICSFILE_CONCEPT.md)
// into a model.LyricList containing a single main Lyrics entry. Returns
// (nil, nil) when the input parses as YAML but does not declare Lyricsfile
// version 1.0.
//
// When the source contains per-word timing via lines[].words[], each word
// becomes a model.Cue with inclusive UTF-8 byte offsets into Line.Value, and
// overlapping lines are attributed to synthetic voice agents via lowest-free
// voice ID assignment so the OpenSubsonic v2 enhanced response can split
// parallel vocals.
func parseLyricsfile(lang string, contents []byte) (LyricList, error) {
	var doc lyricsfileDocument
	dec := yaml.NewDecoder(bytes.NewReader(contents))
	dec.KnownFields(false)
	if err := dec.Decode(&doc); err != nil {
		return nil, fmt.Errorf("not a valid Lyricsfile YAML: %w", err)
	}

	if strings.TrimSpace(doc.Version) != lyricsfileVersion {
		return nil, nil
	}

	// Fall back to the caller's language when the document omits its own, matching
	// the SRT/TTML parsers; normalizeLyricLang yields "xxx" only if both are empty.
	docLang := doc.Metadata.Language
	if strings.TrimSpace(docLang) == "" {
		docLang = lang
	}
	lyrics := Lyrics{
		DisplayArtist: str.SanitizeText(doc.Metadata.Artist),
		DisplayTitle:  str.SanitizeText(doc.Metadata.Title),
		Lang:          normalizeLyricLang(docLang),
		Kind:          LyricKindMain,
	}
	if doc.Metadata.OffsetMs != 0 {
		off := doc.Metadata.OffsetMs
		lyrics.Offset = &off
	}

	if doc.Metadata.Instrumental {
		return LyricList{normalizeLyrics(lyrics)}, nil
	}

	if len(doc.Lines) == 0 {
		lines := buildPlainLyricsfileLines(doc.Plain)
		if len(lines) == 0 {
			return nil, nil
		}
		lyrics.Line = lines
		return LyricList{normalizeLyrics(lyrics)}, nil
	}

	lines, agents := buildLyricsfileLines(doc.Lines)
	lyrics.Line = lines
	lyrics.Agents = agents
	lyrics.Synced = true

	// Normalize the main track before deriving pronunciations: their cues inherit
	// its resolved Start/End, so reordering would break shared timestamps.
	mainLyrics := normalizeLyrics(lyrics)
	result := LyricList{mainLyrics}
	result = append(result, buildLyricsfilePronunciations(doc, mainLyrics)...)
	return result, nil
}

const lyricsfileVersion = "1.0"

type lyricsfileDocument struct {
	Version  string                `yaml:"version"`
	Metadata lyricsfileMetadata    `yaml:"metadata"`
	Lines    []lyricsfileLineEntry `yaml:"lines"`
	Plain    string                `yaml:"plain"`
}

type lyricsfileTransliterationDecl struct {
	ID     string `yaml:"id"`
	System string `yaml:"system"`
}

type lyricsfileMetadata struct {
	Title            string                          `yaml:"title"`
	Artist           string                          `yaml:"artist"`
	Album            string                          `yaml:"album"`
	DurationMs       int64                           `yaml:"duration_ms"`
	OffsetMs         int64                           `yaml:"offset_ms"`
	Language         string                          `yaml:"language"`
	Instrumental     bool                            `yaml:"instrumental"`
	Transliterations []lyricsfileTransliterationDecl `yaml:"transliterations"`
}

type lyricsfileLineEntry struct {
	Text            string                `yaml:"text"`
	StartMs         int64                 `yaml:"start_ms"`
	EndMs           *int64                `yaml:"end_ms"`
	Words           []lyricsfileWordEntry `yaml:"words"`
	Transliteration map[string]string     `yaml:"transliteration"`
}

type lyricsfileWordEntry struct {
	Text            string            `yaml:"text"`
	StartMs         int64             `yaml:"start_ms"`
	EndMs           *int64            `yaml:"end_ms"`
	Transliteration map[string]string `yaml:"transliteration"`
}

// buildLyricsfileLines converts YAML line entries to model.Line entries with
// per-cue AgentIDs assigned by streaming overlap clustering (lowest-free
// voice ID). The Agents slice is emitted only when at least one cue carries
// attribution AND more than one voice is used; otherwise AgentIDs are
// stripped so the wire shape stays simple per the OpenSubsonic spec rule
// "agents should not be emitted without cueLine data".
func buildLyricsfileLines(entries []lyricsfileLineEntry) ([]Line, []Agent) {
	if len(entries) == 0 {
		return nil, nil
	}

	// Resolved end timestamps per entry: explicit end_ms, final word end_ms,
	// then the next entry's start. The last entry's end stays nil when no
	// explicit or word-level end is available.
	ends := make([]*int64, len(entries))
	for i := range entries {
		var nextStart *int64
		if i+1 < len(entries) {
			v := entries[i+1].StartMs
			nextStart = &v
		}
		ends[i] = lyricsfileLineEnd(entries[i], nextStart)
	}

	active := map[int]int64{}
	maxVoice := -1
	anyCues := false
	lines := make([]Line, 0, len(entries))

	for i, entry := range entries {
		for vID, vEnd := range active {
			if vEnd <= entry.StartMs {
				delete(active, vID)
			}
		}

		voiceID := 0
		for {
			if _, busy := active[voiceID]; !busy {
				break
			}
			voiceID++
		}
		if voiceID > maxVoice {
			maxVoice = voiceID
		}

		agentID := fmt.Sprintf("voice-%d", voiceID)
		cues, value := wordsToLineCues(entry, agentID)
		if len(cues) > 0 {
			anyCues = true
		}

		startMs := entry.StartMs
		line := Line{
			Start: &startMs,
			End:   ends[i],
			Value: value,
			Cue:   cues,
		}
		lines = append(lines, line)

		var endMs int64
		if ends[i] != nil {
			endMs = *ends[i]
		} else {
			endMs = entry.StartMs
		}
		active[voiceID] = endMs
	}

	// Monophonic source, or attribution that has nowhere to land: emit no
	// agents and strip per-cue AgentIDs to keep the wire shape simple.
	if maxVoice <= 0 || !anyCues {
		for i := range lines {
			for j := range lines[i].Cue {
				lines[i].Cue[j].AgentID = ""
			}
		}
		return lines, nil
	}

	agents := make([]Agent, 0, maxVoice+1)
	for v := 0; v <= maxVoice; v++ {
		role := "voice"
		if v == 0 {
			role = "main"
		}
		agents = append(agents, Agent{
			ID:   fmt.Sprintf("voice-%d", v),
			Role: role,
		})
	}
	return lines, agents
}

func lyricsfileLineEnd(entry lyricsfileLineEntry, nextStart *int64) *int64 {
	if entry.EndMs != nil {
		v := *entry.EndMs
		return &v
	}
	if len(entry.Words) > 0 {
		lastWord := entry.Words[len(entry.Words)-1]
		if lastWord.EndMs != nil {
			v := *lastWord.EndMs
			return &v
		}
	}
	if nextStart != nil {
		v := *nextStart
		return &v
	}
	return nil
}

func buildPlainLyricsfileLines(plain string) []Line {
	plain = str.SanitizeText(plain)
	rawLines := strings.Split(plain, "\n")
	lines := make([]Line, 0, len(rawLines))
	for _, raw := range rawLines {
		value := strings.TrimSpace(raw)
		if value == "" {
			continue
		}
		lines = append(lines, Line{Value: value})
	}
	return lines
}

// wordsToLineCues converts a Lyricsfile line entry's words[] into model.Cue
// entries with inclusive UTF-8 byte offsets into the reconstructed line
// value. The line value is built from cue text concatenation rather than
// trusting entry.Text, because the Lyricsfile spec only requires word.text
// to "approximate" line.text - byte offsets must always land inside
// Line.Value.
func wordsToLineCues(entry lyricsfileLineEntry, agentID string) ([]Cue, string) {
	if len(entry.Words) == 0 {
		return nil, str.SanitizeText(entry.Text)
	}

	var sb strings.Builder
	for _, w := range entry.Words {
		sb.WriteString(w.Text)
	}
	lineValue := sb.String()

	cues := make([]Cue, len(entry.Words))
	cursor := 0
	for i, w := range entry.Words {
		valueBytes := len(w.Text)
		bs := cursor
		be := bs
		if valueBytes > 0 {
			be = bs + valueBytes - 1
			cursor = be + 1
		}

		s := w.StartMs
		cue := Cue{
			Start:     &s,
			Value:     w.Text,
			ByteStart: bs,
			ByteEnd:   be,
			AgentID:   agentID,
		}
		if w.EndMs != nil {
			e := *w.EndMs
			cue.End = &e
		}
		cues[i] = cue
	}

	for i := 0; i < len(cues)-1; i++ {
		if cues[i].End == nil && cues[i+1].Start != nil {
			v := *cues[i+1].Start
			cues[i].End = &v
		}
	}
	return cues, lineValue
}

// buildLyricsfilePronunciations derives one pronunciation-kind Lyrics per
// declared transliteration system, appended after the main track so the
// OpenSubsonic v2 response exposes each reading as its own track. Every cue
// inherits the matching main cue's Start/End (never invents timings), and words
// lacking a reading for a given system are skipped, so a system's cue array is
// sparse relative to the base. A system with no readings anywhere yields no
// track. mainLyrics must already be normalized: it carries one Line per
// doc.Lines entry and one Cue per word, aligned by index.
func buildLyricsfilePronunciations(doc lyricsfileDocument, mainLyrics Lyrics) LyricList {
	if len(doc.Metadata.Transliterations) == 0 {
		return nil
	}

	out := make(LyricList, 0, len(doc.Metadata.Transliterations))
	for _, decl := range doc.Metadata.Transliterations {
		pronLines := make([]Line, len(mainLyrics.Line))
		for i := range mainLyrics.Line {
			mainLine := mainLyrics.Line[i]
			value, cues := buildPronunciationLine(
				doc.Lines[i].Transliteration[decl.ID],
				doc.Lines[i].Words,
				mainLine.Cue,
				decl.ID,
			)
			pronLines[i] = Line{
				Start: mainLine.Start,
				End:   mainLine.End,
				Value: value,
				Cue:   cues,
			}
		}

		if !hasAnyCue(pronLines) {
			continue
		}

		out = append(out, normalizeLyrics(Lyrics{
			Kind:   LyricKindPronunciation,
			Lang:   normalizeLyricLang(decl.System),
			Synced: mainLyrics.Synced,
			Line:   pronLines,
		}))
	}
	return out
}

// buildPronunciationLine assembles one pronunciation line's value and word cues
// for a single transliteration system. Each cue carries inclusive UTF-8 byte
// offsets into the returned value, honoring the OpenSubsonic songLyrics v2
// contract that cue.byteStart/byteEnd address the parent cueLine.value (not the
// isolated word), so a client recovers the cue text as value[byteStart:byteEnd+1].
//
// Offsets are located against the authored line-level reading by scanning
// forward from the previous cue's end, keeping them monotonic and non-
// overlapping even when untimed text (e.g. a particle with no reading of its
// own) sits between two timed readings. When the line omits an authored reading,
// or its words do not all appear in it verbatim (the Lyricsfile spec only
// requires words to approximate the line), the value is rebuilt by space-joining
// the readings so every offset still lands inside it.
//
// mainCues is the main track's per-word cue slice (one entry per word, aligned by
// index); each emitted cue inherits its matching word's Start/End. Words with no
// reading for this system are skipped, leaving a sparse cue slice.
func buildPronunciationLine(authored string, words []lyricsfileWordEntry, mainCues []Cue, systemID string) (string, []Cue) {
	type reading struct {
		text  string
		start *int64
		end   *int64
	}

	readings := make([]reading, 0, len(words))
	for j, word := range words {
		text := word.Transliteration[systemID]
		if text == "" {
			continue
		}
		readings = append(readings, reading{text: text, start: mainCues[j].Start, end: mainCues[j].End})
	}
	if len(readings) == 0 {
		// No timed readings: expose the authored line value (possibly empty)
		// without cues so the track stays aligned with the main line count.
		return authored, nil
	}

	// Preferred path: address the authored line value, scanning each reading from
	// the previous cue's end so offsets stay ordered and skip untimed gaps.
	if authored != "" {
		cues := make([]Cue, 0, len(readings))
		cursor := 0
		matched := true
		for _, r := range readings {
			rel := strings.Index(authored[cursor:], r.text)
			if rel < 0 {
				matched = false
				break
			}
			start := cursor + rel
			end := start + len(r.text) - 1
			cues = append(cues, Cue{Start: r.start, End: r.end, Value: r.text, ByteStart: start, ByteEnd: end})
			cursor = end + 1
		}
		if matched {
			return authored, cues
		}
	}

	// Fallback: rebuild the line from the readings themselves so byte offsets
	// always land inside the value, space-joining them to preserve word breaks.
	var b strings.Builder
	cues := make([]Cue, 0, len(readings))
	for i, r := range readings {
		if i > 0 {
			b.WriteByte(' ')
		}
		start := b.Len()
		b.WriteString(r.text)
		cues = append(cues, Cue{Start: r.start, End: r.end, Value: r.text, ByteStart: start, ByteEnd: b.Len() - 1})
	}
	return b.String(), cues
}

func hasAnyCue(lines []Line) bool {
	for _, l := range lines {
		if len(l.Cue) > 0 {
			return true
		}
	}
	return false
}
