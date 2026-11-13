--[[
prereqs.lua: per-page prerequisite links, generated at render time from days.yml.

Registered in _quarto.yml (`filters:`), so it runs on every page; it acts only on the pages of
units listed in days.yml (<folder>/lec.qmd, quiz.qmd, exercise.qmd):

- exercise.qmd and quiz.qmd get a short box at the top: "Before this unit:" with links to the
  direct hard prerequisites (the previous unit of the day, the day's hard prerequisites and the
  unit's own), and "See also:" with the soft ones.
- lec.qmd gets one line "Builds on: ..." (the same hard prerequisites) at the end of its
  "About this lecture" slide, before the speaker notes.

A prerequisite that names a day links to the lecture of that day's first unit; one that names a
unit (written "folder/" in days.yml) links to that unit's lecture. Nothing is hand-maintained in
the units: edit days.yml and re-render.
]]

local function stringify(x)
  if x == nil then return nil end
  return pandoc.utils.stringify(x)
end

local function list(x)
  local out = {}
  if x == nil then return out end
  for _, v in ipairs(x) do out[#out + 1] = stringify(v) end
  return out
end

local function dirname(path)
  return path:match('^(.*)[/\\][^/\\]*$') or '.'
end

local function basename(path)
  return path:match('([^/\\]+)$') or path
end

-- Load days.yml once per page (pandoc's markdown reader parses the YAML for us).
local function load_db(root)
  local f = io.open(root .. '/days.yml', 'r')
  if not f then return nil end
  local text = f:read('a')
  f:close()
  local meta = pandoc.read('---\n' .. text .. '\n---\n', 'markdown').meta
  local db = { days = {}, units = {}, day_by_id = {}, unit_by_folder = {} }
  for _, part in ipairs(meta.parts or {}) do
    for _, d in ipairs(part.days or {}) do
      local day = {
        id = stringify(d.id), number = stringify(d.number), title = stringify(d.title),
        hard = list(d.hard), soft = list(d.soft), units = {},
        planned = stringify(d.status) == 'planned',
      }
      db.days[#db.days + 1] = day
      db.day_by_id[day.id] = day
      for k, u in ipairs(d.units or {}) do
        local unit = {
          folder = stringify(u.folder), title = stringify(u.title), day = day, index = k,
          hard = list(u.hard), soft = list(u.soft),
          planned = day.planned or stringify(u.status) == 'planned',
        }
        day.units[#day.units + 1] = unit
        db.units[#db.units + 1] = unit
        db.unit_by_folder[unit.folder] = unit
      end
    end
  end
  return db
end

-- A link (text, folder of the target unit) for a prerequisite reference, or nil.
local function target(db, ref)
  if ref:sub(-1) == '/' then
    local u = db.unit_by_folder[ref:sub(1, -2)]
    if u == nil or u.planned then return nil end
    return { text = u.title .. ' (day ' .. u.day.number .. ')', folder = u.folder }
  end
  local d = db.day_by_id[ref]
  if d == nil or d.planned or d.units[1] == nil then return nil end
  return { text = 'Day ' .. d.number .. ': ' .. d.title, folder = d.units[1].folder }
end

local function collect(db, unit)
  local hard, soft, seen = {}, {}, {}
  local function add(into, t)
    if t and not seen[t.folder .. t.text] then
      seen[t.folder .. t.text] = true
      into[#into + 1] = t
    end
  end
  if unit.index > 1 then
    local prev = unit.day.units[unit.index - 1]
    add(hard, { text = prev.title .. ' (previous unit)', folder = prev.folder })
  end
  for _, r in ipairs(unit.hard) do add(hard, target(db, r)) end
  for _, r in ipairs(unit.day.hard) do add(hard, target(db, r)) end
  for _, r in ipairs(unit.soft) do add(soft, target(db, r)) end
  for _, r in ipairs(unit.day.soft) do add(soft, target(db, r)) end
  return hard, soft
end

local function links(items)
  local out = {}
  for k, t in ipairs(items) do
    if k > 1 then
      out[#out + 1] = pandoc.Str(';')
      out[#out + 1] = pandoc.Space()
    end
    out[#out + 1] = pandoc.Link(pandoc.Inlines(t.text), '../' .. t.folder .. '/lec.qmd')
  end
  out[#out + 1] = pandoc.Str('.')
  return out
end

local function line(label, items)
  local inl = { pandoc.Strong(pandoc.Inlines(label)), pandoc.Space() }
  for _, x in ipairs(links(items)) do inl[#inl + 1] = x end
  return pandoc.Para(inl)
end

function Pandoc(doc)
  local input = quarto and quarto.doc and quarto.doc.input_file
  if input == nil then return nil end
  local page = basename(input):gsub('%.qmd$', '')
  if page ~= 'lec' and page ~= 'quiz' and page ~= 'exercise' then return nil end
  local folder_path = dirname(input)
  local folder = basename(folder_path)
  local root = (quarto.project and quarto.project.directory) or dirname(folder_path)
  local db = load_db(root)
  if db == nil then return nil end
  local unit = db.unit_by_folder[folder]
  if unit == nil then return nil end
  local hard, soft = collect(db, unit)
  if #hard == 0 and #soft == 0 then return nil end

  if page == 'lec' then
    if #hard == 0 then return nil end
    -- Find the "About this lecture" slide; insert before its notes, else at its end.
    local blocks = doc.blocks
    local start
    for i, b in ipairs(blocks) do
      if b.t == 'Header' and b.level <= 2 and stringify(b.content) == 'About this lecture' then
        start = i
        break
      end
    end
    if start == nil then return nil end
    local stop = #blocks + 1
    for i = start + 1, #blocks do
      local b = blocks[i]
      if (b.t == 'Header' and b.level <= 2) or b.t == 'HorizontalRule' then
        stop = i
        break
      end
      if b.t == 'Div' and b.classes:includes('notes') then
        stop = i
        break
      end
    end
    local div = pandoc.Div({ line('Builds on:', hard) },
      pandoc.Attr('', { 'builds-on' }, { style = 'font-size: 0.6em; margin-top: 0.4em;' }))
    blocks:insert(stop, div)
    return doc
  end

  local content = {}
  if #hard > 0 then content[#content + 1] = line('Before this unit:', hard) end
  if #soft > 0 then content[#content + 1] = line('See also:', soft) end
  doc.blocks:insert(1, pandoc.Div(content, pandoc.Attr('', { 'prereq-box' })))
  return doc
end
